//! Basalt semantic-facts plugin for Rust (LSP-based).
//!
//! Spawns `rust-analyzer` via the host's LSP subprocess API and queries it
//! for import, implementation, and dependency relationships.
//!
//! Falls back to regex-based extraction when rust-analyzer is unavailable.

use basalt_plugin_sdk::facts::{
    AsyncBoundaryKind, ControlFlowKind, IOKind, MutationKind, MutationTarget,
    SemanticFact, SymbolKind, serialize_facts,
};
use basalt_plugin_sdk::prelude::*;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

// ── Host imports ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn basalt_spawn_lsp(cmd_ptr: *const u8, cmd_len: usize, root_ptr: *const u8, root_len: usize) -> i32;
    fn basalt_lsp_write(handle: i32, buf_ptr: *const u8, buf_len: usize) -> i32;
    fn basalt_lsp_read(handle: i32, out_ptr: *mut u8, out_cap: usize) -> i32;
    fn basalt_lsp_stop(handle: i32);
    fn basalt_sleep_ms(ms: u32);
    fn basalt_log(level: i32, msg_ptr: *const u8, msg_len: usize);
}

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_spawn_lsp(_: *const u8, _: usize, _: *const u8, _: usize) -> i32 { -1 }
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_lsp_write(_: i32, _: *const u8, _: usize) -> i32 { 0 }
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_lsp_read(_: i32, _: *mut u8, _: usize) -> i32 { 0 }
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_lsp_stop(_: i32) {}
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_sleep_ms(_: u32) {}
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_log(_: i32, _: *const u8, _: usize) {}
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_capability_invoke(_: i32, _: i32, _: i32, _: i32) -> i64 { 0 }
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_capability_free_response(_: i32) {}
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
unsafe extern "C" fn basalt_capability_copy_response(_: i32, _: i32, _: i32) -> i32 { 0 }

fn log_info(msg: &str) {
    unsafe { basalt_log(1, msg.as_ptr(), msg.len()); }
}

// ── Plugin metadata ───────────────────────────────────────────────────────────

basalt_plugin_meta! {
    name:              "semantic-facts-rust",
    version:           "0.1.0",
    hook_flags:        CAP_CAPABILITY_HANDLE | CAP_REVIEW_ACTIONS | CAP_BUILD_SYSTEM | CAP_TEST_RUNNER | CAP_WASM_VERIFY,
    provides:          "semantic-facts@rust/v1",
    requires:          "",
    optional_requires: "",
    file_globs:        "**/*.rs",
    activates_on:      "**/Cargo.toml\nCargo.toml",
    activation_events: "",
}

// ── LSP JSON-RPC helpers ─────────────────────────────────────────────────────

static mut REQUEST_ID: u32 = 1;
static mut LSP_HANDLE: i32 = -1;
static mut LSP_INITIALIZED: bool = false;

fn next_id() -> u32 {
    unsafe {
        let id = REQUEST_ID;
        REQUEST_ID += 1;
        id
    }
}

/// Build a Content-Length framed LSP message. Returns (bytes, request_id).
fn lsp_message(method: &str, params: &str) -> (Vec<u8>, u32) {
    let id = next_id();
    let body = format!(
        r#"{{"jsonrpc":"2.0","id":{},"method":"{}","params":{}}}"#,
        id, method, params
    );
    (format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes(), id)
}

fn lsp_notification(method: &str, params: &str) -> Vec<u8> {
    let body = format!(
        r#"{{"jsonrpc":"2.0","method":"{}","params":{}}}"#,
        method, params
    );
    format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
}

/// Write bytes to the LSP stdin.
fn lsp_write(handle: i32, data: &[u8]) -> bool {
    let written = unsafe { basalt_lsp_write(handle, data.as_ptr(), data.len()) };
    written >= 0
}

/// Read all available bytes from LSP stdout (non-blocking, with retries).
fn lsp_read_all(handle: i32) -> Vec<u8> {
    let mut buf = vec![0u8; 256 * 1024];
    let mut result = Vec::new();
    for _ in 0..50 {
        let n = unsafe { basalt_lsp_read(handle, buf.as_mut_ptr(), buf.len()) };
        if n > 0 {
            result.extend_from_slice(&buf[..n as usize]);
        } else {
            unsafe { basalt_sleep_ms(20); }
        }
        // Check if we have a complete message
        if let Some(idx) = find_content_length_end(&result) {
            if idx < result.len() {
                break;
            }
        }
    }
    result
}

/// Read a specific LSP response by request ID, discarding notifications and
/// other responses. Returns the full raw message (header + body) or empty on timeout.
/// Used during readiness probing where unsolicited notifications may arrive first.
fn lsp_read_response(handle: i32, expected_id: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 256 * 1024];
    let mut accumulator = Vec::new();
    // ~60s total: 120 iterations × 500ms sleep
    for _ in 0..120 {
        let n = unsafe { basalt_lsp_read(handle, buf.as_mut_ptr(), buf.len()) };
        if n > 0 {
            accumulator.extend_from_slice(&buf[..n as usize]);
        } else {
            unsafe { basalt_sleep_ms(500); }
        }

        // Process complete messages from the accumulator
        while let Some(header_end) = find_content_length_end(&accumulator) {
            let content_len = match parse_content_length(&accumulator) {
                Some(len) => len,
                None => break,
            };
            let total_msg = header_end + content_len;
            if accumulator.len() < total_msg {
                break; // incomplete body, wait for more data
            }

            let body_bytes = &accumulator[header_end..total_msg];
            let body = String::from_utf8_lossy(body_bytes);

            // Check if this is the response we're looking for
            if let Some(id) = json_extract_i32(&body, "id") {
                if id as u32 == expected_id {
                    // Found it — return the complete message
                    let msg = accumulator[..total_msg].to_vec();
                    accumulator.drain(..total_msg);
                    return msg;
                }
                // Different request ID — discard and continue
            }
            // No "id" field (notification) — discard and continue
            accumulator.drain(..total_msg);
        }
    }
    Vec::new()
}

fn find_content_length_end(data: &[u8]) -> Option<usize> {
    let marker = b"\r\n\r\n";
    for i in 0..data.len().saturating_sub(3) {
        if &data[i..i + 4] == marker {
            return Some(i + 4);
        }
    }
    None
}

/// Parse Content-Length from an LSP response header.
fn parse_content_length(data: &[u8]) -> Option<usize> {
    let header_end = find_content_length_end(data)?;
    let header = &data[..header_end.saturating_sub(4)];
    for line in String::from_utf8_lossy(header).lines() {
        if let Some(val) = line.strip_prefix("Content-Length: ") {
            return val.trim().parse().ok();
        }
    }
    None
}

/// Extract a JSON string field value (simple parser, no serde needed).
fn json_extract_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    if rest.starts_with('"') {
        let end = rest[1..].find('"')?;
        Some(rest[1..=end].to_string())
    } else {
        None
    }
}

/// Extract an integer field value.
fn json_extract_i32(json: &str, key: &str) -> Option<i32> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract an array of Location objects from a textDocument/implementation result.
/// Each Location has `uri` and `range.start.line`/`range.start.character`.
fn json_extract_locations(json: &str) -> Vec<(String, u32, u32)> {
    let mut locations = Vec::new();
    // Find all "uri":"..." pairs
    let mut pos = 0;
    while let Some(uri_start) = json[pos..].find("\"uri\":\"") {
        let abs = pos + uri_start + 7;
        if let Some(uri_end) = json[abs..].find('"') {
            let uri = json[abs..abs + uri_end].to_string();
            // Look for the next line/character
            let search_region = &json[abs + uri_end..];
            let line = search_region.find("\"line\":")
                .and_then(|i| {
                    let v = &search_region[i + 7..];
                    let end = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
                    v[..end].parse::<u32>().ok()
                })
                .unwrap_or(0);
            let character = search_region.find("\"character\":")
                .and_then(|i| {
                    let v = &search_region[i + 12..];
                    let end = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
                    v[..end].parse::<u32>().ok()
                })
                .unwrap_or(0);
            locations.push((uri, line, character));
            pos = abs + uri_end + 1;
        } else {
            break;
        }
    }
    locations
}

// ── LSP lifecycle ─────────────────────────────────────────────────────────────

/// Poll rust-analyzer with `workspace/symbol` until it responds, indicating
/// the workspace index is ready. Returns true if indexing completed within timeout.
fn wait_for_indexing(handle: i32) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ALREADY_READY: AtomicBool = AtomicBool::new(false);
    if ALREADY_READY.load(Ordering::Relaxed) {
        return true;
    }

    let mut delay_ms: u32 = 500;
    const MAX_DELAY_MS: u32 = 5_000;
    const MAX_ATTEMPTS: u32 = 60;

    for attempt in 1..=MAX_ATTEMPTS {
        // Send workspace/symbol probe — requires full index to respond
        let (msg, id) = lsp_message("workspace/symbol", r#"{"query":""}"#);
        if !lsp_write(handle, &msg) {
            unsafe { basalt_sleep_ms(delay_ms); }
            continue;
        }

        let response = lsp_read_response(handle, id);
        if response.is_empty() {
            // Timeout or no response — still indexing
            let log_msg = format!(
                "[semantic-facts-rust] Waiting for rust-analyzer indexing... (attempt {}/{})",
                attempt, MAX_ATTEMPTS
            );
            log_info(&log_msg);
            unsafe { basalt_sleep_ms(delay_ms); }
            delay_ms = (delay_ms * 2).min(MAX_DELAY_MS);
            continue;
        }

        // Got a response — check if it's an error (still indexing) or success (ready)
        let body_start = find_content_length_end(&response).unwrap_or(0);
        if body_start >= response.len() {
            unsafe { basalt_sleep_ms(delay_ms); }
            continue;
        }
        let body = String::from_utf8_lossy(&response[body_start..]);

        // An error response means rust-analyzer isn't ready yet
        if body.contains(r#""error""#) {
            let log_msg = format!(
                "[semantic-facts-rust] Waiting for rust-analyzer indexing... (attempt {}/{}, got error)",
                attempt, MAX_ATTEMPTS
            );
            log_info(&log_msg);
            unsafe { basalt_sleep_ms(delay_ms); }
            delay_ms = (delay_ms * 2).min(MAX_DELAY_MS);
            continue;
        }

        // Success response (even empty array means the index is available)
        let log_msg = format!(
            "[semantic-facts-rust] rust-analyzer indexing complete (after {} attempts)",
            attempt
        );
        log_info(&log_msg);
        ALREADY_READY.store(true, Ordering::Relaxed);
        return true;
    }

    log_info("[semantic-facts-rust] rust-analyzer indexing timeout — proceeding with partial index");
    false
}

fn lsp_stop(handle: i32) {
    // Send proper LSP shutdown sequence before killing
    let (shutdown, _) = lsp_message("shutdown", "null");
    let _ = lsp_write(handle, &shutdown);
    // Give it a moment to process shutdown, then exit notification
    unsafe { basalt_sleep_ms(100); }
    let exit = lsp_notification("exit", "null");
    let _ = lsp_write(handle, &exit);
    unsafe { basalt_sleep_ms(100); }
    unsafe { basalt_lsp_stop(handle); }
}

fn ensure_lsp_running(workspace_root: &str) -> Option<i32> {
    unsafe {
        if LSP_HANDLE >= 0 {
            return Some(LSP_HANDLE);
        }
    }

    // Don't retry spawning after the first failure
    use std::sync::atomic::{AtomicBool, Ordering};
    static SPAWN_FAILED: AtomicBool = AtomicBool::new(false);
    if SPAWN_FAILED.load(Ordering::Relaxed) {
        return None;
    }

    // Try to spawn rust-analyzer (try common name variants)
    let root_bytes = workspace_root.as_bytes();
    let names = ["rust-analyzer\0", "rust_analyzer\0"];
    let mut handle = -1i32;
    for name in &names {
        handle = unsafe {
            basalt_spawn_lsp(
                name.as_ptr(),
                name.len(),
                root_bytes.as_ptr(),
                root_bytes.len(),
            )
        };
        if handle >= 0 {
            break;
        }
    }

    if handle < 0 {
        SPAWN_FAILED.store(true, Ordering::Relaxed);
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            log_info("[semantic-facts-rust] rust-analyzer not found on PATH — install it for richer cross-file edges:");
            log_info("[semantic-facts-rust]   rustup component add rust-analyzer");
            log_info("[semantic-facts-rust] Falling back to regex extraction (use/impl only)");
        }
        return None;
    }

    unsafe { LSP_HANDLE = handle; }

    // Send initialize
    let init_params = r#"{"processId":null,"capabilities":{"textDocument":{"implementation":{"dynamicRegistration":false},"definition":{"dynamicRegistration":false},"references":{"dynamicRegistration":false}}}}"#;
    let (msg, init_id) = lsp_message("initialize", init_params);
    if !lsp_write(handle, &msg) {
        log_info("[semantic-facts-rust] failed to write initialize");
        lsp_stop(handle);
        unsafe { LSP_HANDLE = -1; }
        return None;
    }

    // Read response (blocking, ID-aware — may get log notifications first)
    let response = lsp_read_response(handle, init_id);
    if response.is_empty() {
        log_info("[semantic-facts-rust] no initialize response");
        lsp_stop(handle);
        unsafe { LSP_HANDLE = -1; }
        return None;
    }

    // Send initialized notification
    let initialized = lsp_notification("initialized", "{}");
    let _ = lsp_write(handle, &initialized);
    unsafe { LSP_INITIALIZED = true; }

    log_info("[semantic-facts-rust] rust-analyzer started — waiting for indexing...");

    // Wait for rust-analyzer to finish indexing before accepting queries
    wait_for_indexing(handle);

    Some(handle)
}

fn lsp_did_open(handle: i32, uri: &str, language_id: &str, text: &str) {
    let params = format!(
        r#"{{"textDocument":{{"uri":"{}","languageId":"{}","version":0,"text":"{}"}}}}"#,
        uri,
        language_id,
        json_escape_string(text)
    );
    let msg = lsp_notification("textDocument/didOpen", &params);
    let _ = lsp_write(handle, &msg);
}

/// Escape a string for use inside a JSON string value.
/// Handles all control characters (U+0000..U+001F) as required by RFC 8259.
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 4);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            _ => out.push(c),
        }
    }
    out
}

fn lsp_query_implementation(handle: i32, uri: &str, line: u32, character: u32) -> Vec<(String, u32, u32)> {
    let params = format!(
        r#"{{"textDocument":{{"uri":"{}"}},"position":{{"line":{},"character":{}}}}}"#,
        uri, line, character
    );
    let (msg, id) = lsp_message("textDocument/implementation", &params);
    if !lsp_write(handle, &msg) {
        return Vec::new();
    }
    let response = lsp_read_response(handle, id);
    if response.is_empty() {
        return Vec::new();
    }
    let body_start = find_content_length_end(&response).unwrap_or(0);
    if body_start >= response.len() {
        return Vec::new();
    }
    let body = String::from_utf8_lossy(&response[body_start..]);
    json_extract_locations(&body)
}

fn lsp_query_references(handle: i32, uri: &str, line: u32, character: u32) -> Vec<(String, u32, u32)> {
    let params = format!(
        r#"{{"textDocument":{{"uri":"{}"}},"position":{{"line":{},"character":{}}},"context":{{"includeDeclaration":true}}}}"#,
        uri, line, character
    );
    let (msg, id) = lsp_message("textDocument/references", &params);
    if !lsp_write(handle, &msg) {
        return Vec::new();
    }
    let response = lsp_read_response(handle, id);
    if response.is_empty() {
        return Vec::new();
    }
    let body_start = find_content_length_end(&response).unwrap_or(0);
    if body_start >= response.len() {
        return Vec::new();
    }
    let body = String::from_utf8_lossy(&response[body_start..]);
    json_extract_locations(&body)
}

// ── Regex fallback ────────────────────────────────────────────────────────────

fn extract_imports_regex(src: &str) -> Vec<SemanticFact> {
    let mut facts = Vec::new();
    let src_bytes = src.as_bytes();

    for line in src.lines() {
        let trimmed = line.trim_start();
        // Rust: use foo::bar::baz;
        if trimmed.starts_with("use ") && trimmed.contains("::") {
            let offset = (line.as_ptr() as usize - src.as_ptr() as usize) as u32;
            let length = line.len() as u32;
            let path_part = trimmed
                .strip_prefix("use ")
                .unwrap_or(trimmed)
                .trim_end_matches(';')
                .trim();
            // Handle grouped imports: use foo::{bar, baz}
            if let Some(group_start) = path_part.find('{') {
                let base_path = path_part[..group_start].trim().trim_end_matches("::");
                for item in path_part[group_start + 1..].trim_end_matches('}').split(',') {
                    let item = item.trim().split_whitespace().next().unwrap_or("");
                    if !item.is_empty() && !item.starts_with("super") && !item.starts_with("self") && !item.starts_with("crate") {
                        facts.push(SemanticFact::ImportModule {
                            offset,
                            length,
                            module_path: format!("{}::{}", base_path, item),
                            alias: None,
                        });
                    }
                }
            } else {
                // Handle `use foo as bar`
                let (full_path, alias) = if let Some(as_pos) = path_part.find(" as ") {
                    let p = path_part[..as_pos].trim();
                    let a = path_part[as_pos + 4..].trim();
                    (p.to_string(), Some(a.to_string()))
                } else {
                    (path_part.to_string(), None)
                };
                facts.push(SemanticFact::ImportModule {
                    offset,
                    length,
                    module_path: full_path,
                    alias,
                });
            }
        }
    }

    facts
}

fn extract_impls_regex(src: &str) -> Vec<SemanticFact> {
    let mut facts = Vec::new();

    for line in src.lines() {
        let trimmed = line.trim_start();
        // Rust: impl Trait for Type
        if trimmed.starts_with("impl<") || trimmed.starts_with("impl ") {
            let offset = (line.as_ptr() as usize - src.as_ptr() as usize) as u32;
            let length = line.len() as u32;
            // Try to match `impl TraitName for TypeName`
            if let Some(for_pos) = trimmed.find(" for ") {
                let trait_part = &trimmed[4..for_pos];
                let type_part = trimmed[for_pos + 5..].trim().trim_end_matches('{').trim();
                // Extract just the trait name (strip generics)
                let trait_name = trait_part.split('<').next().unwrap_or(trait_part).trim();
                // Extract just the type name (strip generics)
                let type_name = type_part.split('<').next().unwrap_or(type_part).trim();

                if !trait_name.is_empty() && !type_name.is_empty() {
                    facts.push(SemanticFact::Implements {
                        offset,
                        type_name: type_name.to_string(),
                        contract: trait_name.to_string(),
                    });
                }
            }
        }
    }

    facts
}

// ── AST Visitor ───────────────────────────────────────────────────────────────

fn span_to_offset_len(span: proc_macro2::Span) -> (u32, u32) {
    let range = span.byte_range();
    (range.start as u32, range.len() as u32)
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(tp) => path_to_string(&tp.path),
        _ => String::new(),
    }
}

fn expr_to_name(expr: &syn::Expr) -> String {
    match expr {
        syn::Expr::Path(ep) => path_to_string(&ep.path),
        syn::Expr::Field(ef) => {
            let base = expr_to_name(&ef.base);
            match &ef.member {
                syn::Member::Named(id) => format!("{}.{}", base, id),
                syn::Member::Unnamed(idx) => format!("{}.{}", base, idx.index),
            }
        }
        _ => "target".to_string(),
    }
}

fn extract_use_tree(
    tree: &syn::UseTree,
    prefix: String,
    offset: u32,
    length: u32,
    facts: &mut Vec<SemanticFact>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            let next_prefix = if prefix.is_empty() {
                p.ident.to_string()
            } else {
                format!("{}::{}", prefix, p.ident)
            };
            extract_use_tree(&p.tree, next_prefix, offset, length, facts);
        }
        syn::UseTree::Name(n) => {
            let module_path = if prefix.is_empty() {
                n.ident.to_string()
            } else {
                format!("{}::{}", prefix, n.ident)
            };
            facts.push(SemanticFact::ImportModule {
                offset,
                length,
                module_path,
                alias: None,
            });
        }
        syn::UseTree::Rename(r) => {
            let module_path = if prefix.is_empty() {
                r.ident.to_string()
            } else {
                format!("{}::{}", prefix, r.ident)
            };
            facts.push(SemanticFact::ImportModule {
                offset,
                length,
                module_path,
                alias: Some(r.rename.to_string()),
            });
        }
        syn::UseTree::Glob(_) => {
            if !prefix.is_empty() {
                facts.push(SemanticFact::ImportModule {
                    offset,
                    length,
                    module_path: format!("{}::*", prefix),
                    alias: None,
                });
            }
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                extract_use_tree(item, prefix.clone(), offset, length, facts);
            }
        }
    }
}

pub struct AstFactVisitor {
    pub facts: Vec<SemanticFact>,
    /// Enclosing scope stack (modules, impl self-types, outer fns).
    /// Joined with `::` to build qualified names for declares.
    scope: Vec<String>,
    /// Enclosing `impl` self-types (innermost last). Used to qualify
    /// `self.` / `Self::` call receivers.
    impl_stack: Vec<String>,
    /// Module name derived from the file stem (`foo.rs` → `foo`).
    /// `None` for crate roots (`main.rs`/`lib.rs`).
    module: Option<String>,
}

impl AstFactVisitor {
    pub fn new() -> Self {
        Self { facts: Vec::new(), scope: Vec::new(), impl_stack: Vec::new(), module: None }
    }

    pub fn set_module(&mut self, module: Option<String>) {
        self.module = module;
    }

    /// Qualified name for `name` declared in the current scope:
    /// `module::Scope::name`. `None` only when there is no scope context
    /// at all (bare top-level item in a crate root).
    fn qualified(&self, name: &str) -> Option<String> {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(ref m) = self.module {
            parts.push(m.as_str());
        }
        parts.extend(self.scope.iter().map(|s| s.as_str()));
        parts.push(name);
        if parts.len() <= 1 {
            None
        } else {
            Some(parts.join("::"))
        }
    }

    fn current_impl_type(&self) -> Option<&str> {
        self.impl_stack.last().map(|s| s.as_str())
    }
}

/// Module name for a file path (`foo.rs` → `Some("foo")`). Crate roots
/// (`main.rs`/`lib.rs`) and unknown files yield `None`; `mod.rs` resolves
/// to its parent directory name.
fn module_name_for_path(path: &str) -> Option<String> {
    let norm = path.replace('\\', "/");
    let file = norm.rsplit('/').next().unwrap_or(&norm);
    let stem = file.strip_suffix(".rs").unwrap_or(file);
    match stem {
        "main" | "lib" | "" => None,
        "mod" => {
            let mut parts = norm.rsplit('/');
            parts.next();
            parts
                .next()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty() && s != ".")
        }
        s => Some(s.to_string()),
    }
}

fn classify_io(callee: &str) -> Option<IOKind> {
    let lower = callee.to_ascii_lowercase();
    if lower.contains("reqwest")
        || lower.contains("fetch")
        || lower.contains("hyper")
        || lower.contains("ureq")
        || lower.contains("tcpstream")
        || lower.contains("udpsocket")
        || lower.ends_with("::get")
        || lower.ends_with("::post")
    {
        Some(IOKind::Network)
    } else if lower.contains("read_to_string")
        || lower.contains("read_exact")
        || lower.contains("file::open")
        || lower.contains("fs::read")
    {
        Some(IOKind::FileRead)
    } else if lower.contains("write_all")
        || lower.contains("file::create")
        || lower.contains("fs::write")
        || lower.contains("persist")
    {
        Some(IOKind::FileWrite)
    } else if lower.contains("println")
        || lower.contains("eprintln")
        || lower.contains("stdout")
        || lower.contains("stderr")
        || lower.contains("stdin")
    {
        Some(IOKind::StdIO)
    } else {
        None
    }
}

impl<'ast> Visit<'ast> for AstFactVisitor {
    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::DeclareSymbol {
            kind: SymbolKind::Struct,
            offset,
            length,
            name: node.ident.to_string(),
            qualified_name: self.qualified(&node.ident.to_string()),
        });
        visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::DeclareSymbol {
            kind: SymbolKind::Enum,
            offset,
            length,
            name: node.ident.to_string(),
            qualified_name: self.qualified(&node.ident.to_string()),
        });
        visit::visit_item_enum(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::DeclareSymbol {
            kind: SymbolKind::Trait,
            offset,
            length,
            name: node.ident.to_string(),
            qualified_name: self.qualified(&node.ident.to_string()),
        });
        visit::visit_item_trait(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let (offset, length) = span_to_offset_len(node.span());
        let name = node.sig.ident.to_string();
        self.facts.push(SemanticFact::DeclareSymbol {
            kind: SymbolKind::Function,
            offset,
            length,
            name: name.clone(),
            qualified_name: self.qualified(&name),
        });
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Function,
            offset,
            length,
        });
        self.scope.push(name);
        visit::visit_item_fn(self, node);
        self.scope.pop();
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let (offset, length) = span_to_offset_len(node.span());
        let name = node.sig.ident.to_string();
        self.facts.push(SemanticFact::DeclareSymbol {
            kind: SymbolKind::Method,
            offset,
            length,
            name: name.clone(),
            qualified_name: self.qualified(&name),
        });
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Function,
            offset,
            length,
        });
        visit::visit_impl_item_fn(self, node);
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let self_ty = type_to_string(&node.self_ty);
        let pushed = !self_ty.is_empty();
        if pushed {
            self.scope.push(self_ty.clone());
            self.impl_stack.push(self_ty);
        }
        if let Some((_, ref trait_path, _)) = node.trait_ {
            let trait_name = path_to_string(trait_path);
            let type_name = type_to_string(&node.self_ty);
            if !trait_name.is_empty() && !type_name.is_empty() {
                let (offset, _length) = span_to_offset_len(node.span());
                self.facts.push(SemanticFact::Implements {
                    offset,
                    type_name,
                    contract: trait_name,
                });
            }
        }
        visit::visit_item_impl(self, node);
        if pushed {
            self.impl_stack.pop();
            self.scope.pop();
        }
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let name = node.ident.to_string();
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::DeclareSymbol {
            kind: SymbolKind::Module,
            offset,
            length,
            name: name.clone(),
            qualified_name: self.qualified(&name),
        });
        self.scope.push(name);
        visit::visit_item_mod(self, node);
        self.scope.pop();
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        let (offset, length) = span_to_offset_len(node.span());
        extract_use_tree(&node.tree, String::new(), offset, length, &mut self.facts);
        visit::visit_item_use(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Conditional,
            offset,
            length,
        });
        visit::visit_expr_if(self, node);
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Loop,
            offset,
            length,
        });
        visit::visit_expr_loop(self, node);
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Loop,
            offset,
            length,
        });
        visit::visit_expr_while(self, node);
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Loop,
            offset,
            length,
        });
        visit::visit_expr_for_loop(self, node);
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::ControlFlowEnter {
            kind: ControlFlowKind::Switch,
            offset,
            length,
        });
        visit::visit_expr_match(self, node);
        self.facts.push(SemanticFact::ControlFlowExit {
            offset: offset + length.saturating_sub(1),
            length: 1,
        });
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        let (offset, length) = span_to_offset_len(node.span());
        let mut callee = match &*node.func {
            syn::Expr::Path(ep) => path_to_string(&ep.path),
            _ => String::new(),
        };
        // `Self::assoc()` inside `impl Ty` → `Ty::assoc` for exact matching.
        if let Some(ty) = self.current_impl_type() {
            let ty = ty.to_string();
            if callee == "Self" {
                callee = ty;
            } else if let Some(rest) = callee.strip_prefix("Self::") {
                callee = format!("{ty}::{rest}");
            }
        }
        if !callee.is_empty() {
            self.facts.push(SemanticFact::Calls {
                caller_offset: offset,
                caller_length: length,
                callee: callee.clone(),
            });
            if let Some(io_kind) = classify_io(&callee) {
                self.facts.push(SemanticFact::IOOperation {
                    io_kind,
                    offset,
                    length,
                    descriptor: Some(callee),
                });
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let (offset, length) = span_to_offset_len(node.span());
        let method = node.method.to_string();
        // `self.foo()` inside `impl Ty` → `Ty::foo` for exact matching.
        // Other receivers stay bare — core falls back to same-file matching.
        let is_self = matches!(&*node.receiver, syn::Expr::Path(ep) if ep.path.is_ident("self"));
        let callee = if is_self {
            if let Some(ty) = self.current_impl_type() {
                format!("{ty}::{method}")
            } else {
                method.clone()
            }
        } else {
            method.clone()
        };
        self.facts.push(SemanticFact::Calls {
            caller_offset: offset,
            caller_length: length,
            callee: callee.clone(),
        });
        if let Some(io_kind) = classify_io(&callee) {
            self.facts.push(SemanticFact::IOOperation {
                io_kind,
                offset,
                length,
                descriptor: Some(callee),
            });
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let (offset, length) = span_to_offset_len(node.span());
        let macro_name = path_to_string(&node.path);
        if !macro_name.is_empty() {
            if let Some(io_kind) = classify_io(&macro_name) {
                self.facts.push(SemanticFact::IOOperation {
                    io_kind,
                    offset,
                    length,
                    descriptor: Some(macro_name),
                });
            }
        }
        visit::visit_macro(self, node);
    }

    fn visit_expr_await(&mut self, node: &'ast syn::ExprAwait) {
        let (offset, length) = span_to_offset_len(node.span());
        self.facts.push(SemanticFact::AsyncBoundary {
            boundary_kind: AsyncBoundaryKind::Await,
            offset,
            length,
        });
        visit::visit_expr_await(self, node);
    }

    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        let (offset, length) = span_to_offset_len(node.span());
        let target_name = expr_to_name(&node.left);
        self.facts.push(SemanticFact::Mutation {
            target: MutationTarget::Variable,
            kind: MutationKind::StateWrite,
            offset,
            length,
            name: target_name,
        });
        visit::visit_expr_assign(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        let is_compound_assign = matches!(
            node.op,
            syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
                | syn::BinOp::BitXorAssign(_)
                | syn::BinOp::BitAndAssign(_)
                | syn::BinOp::BitOrAssign(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        );
        if is_compound_assign {
            let (offset, length) = span_to_offset_len(node.span());
            let target_name = expr_to_name(&node.left);
            self.facts.push(SemanticFact::Mutation {
                target: MutationTarget::Variable,
                kind: MutationKind::StateWrite,
                offset,
                length,
                name: target_name,
            });
        }
        visit::visit_expr_binary(self, node);
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

fn extract_semantic_facts_for_file(src: &[u8], path: &str, workspace_root: &str) -> Vec<SemanticFact> {
    let text = String::from_utf8_lossy(src);
    let mut facts = Vec::new();

    // Try LSP first — use host-provided workspace root
    let root = if workspace_root.is_empty() {
        infer_workspace_root(path)
    } else {
        workspace_root.to_string()
    };
    let lsp_available = ensure_lsp_running(&root).is_some();

    if lsp_available {
        let handle = unsafe { LSP_HANDLE };
        let uri = path_to_uri(path);
        lsp_did_open(handle, &uri, "rust", &text);
    }

    // Try AST extraction first; fall back to regex if syntax parsing fails
    if let Ok(syntax_tree) = syn::parse_file(&text) {
        let mut visitor = AstFactVisitor::new();
        visitor.set_module(module_name_for_path(path));
        visitor.visit_file(&syntax_tree);
        facts.extend(visitor.facts);
    } else {
        facts.extend(extract_imports_regex(&text));
        facts.extend(extract_impls_regex(&text));
    }

    // LSP-enhanced cross-file edge extraction
    if lsp_available {
        let handle = unsafe { LSP_HANDLE };
        let uri = path_to_uri(path);
        let mut lsp_facts = Vec::new();

        // Query textDocument/implementation for each Implements fact to find cross-file implementations
        for fact in &facts {
            if let SemanticFact::Implements { offset, .. } = fact {
                let (line, col) = byte_to_line_col(&text, *offset);
                let locations = lsp_query_implementation(handle, &uri, line, col);
                for (loc_uri, _loc_line, _loc_col) in &locations {
                    let loc_path = uri_to_path(loc_uri);
                    if loc_path.is_empty() || loc_path == path {
                        continue;
                    }
                    lsp_facts.push(SemanticFact::DependsOn {
                        offset: *offset,
                        length: 0,
                        dependency: loc_path,
                    });
                }
            }
        }

        // Extract function definitions and query references for cross-file CALLS
        let func_offsets = extract_function_positions(&text);
        for (offset, _length, func_name) in &func_offsets {
            let (line, col) = byte_offset_to_line_col(&text, *offset);
            let locations = lsp_query_references(handle, &uri, line, col);
            for (loc_uri, _loc_line, _loc_col) in &locations {
                let loc_path = uri_to_path(loc_uri);
                if !loc_path.is_empty() && loc_path != path {
                    lsp_facts.push(SemanticFact::DependsOn {
                        offset: *offset,
                        length: 0,
                        dependency: format!("{}::{}", loc_path, func_name),
                    });
                }
            }
        }

        facts.extend(lsp_facts);
    }

    facts
}

fn infer_workspace_root(file_path: &str) -> String {
    // Normalize Windows backslashes to forward slashes for wasm32-unknown-unknown
    // where std::path::Path does not recognize '\' as a separator.
    let normalized = file_path.replace('\\', "/");
    let path = std::path::Path::new(&normalized);
    let mut current = path.parent().unwrap_or(path);
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            // Convert back to Windows-style if the original path used backslashes
            let result = current.to_string_lossy().to_string();
            if file_path.contains('\\') {
                return result.replace('/', "\\");
            }
            return result;
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => return String::new(),
        }
    }
}

fn path_to_uri(path: &str) -> String {
    // LSP URIs always use forward slashes, even on Windows
    format!("file:///{}", path.replace('\\', "/"))
}

fn uri_to_path(uri: &str) -> String {
    uri.strip_prefix("file:///")
        .or_else(|| uri.strip_prefix("file://"))
        .unwrap_or(uri)
        .replace('/', if uri.contains('\\') { "\\" } else { "/" })
}

fn byte_to_line_col(text: &str, byte_offset: u32) -> (u32, u32) {
    let mut line = 0u32;
    let mut col = 0u32;
    let mut byte_pos = 0u32;
    for ch in text.chars() {
        if byte_pos >= byte_offset {
            break;
        }
        byte_pos += ch.len_utf8() as u32;
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn byte_offset_to_line_col(text: &str, byte_offset: u32) -> (u32, u32) {
    byte_to_line_col(text, byte_offset)
}

/// Extract (byte_offset, length, name) for function definitions.
fn extract_function_positions(text: &str) -> Vec<(u32, u32, String)> {
    let mut results = Vec::new();
    let mut byte_offset = 0u32;

    for line in text.lines() {
        let trimmed = line.trim_start();
        // Strip optional visibility: pub, pub(crate), pub(super), pub(in path)
        let after_vis = if let Some(rest) = trimmed.strip_prefix("pub") {
            let rest = rest.trim_start();
            if rest.starts_with('(') {
                // pub(crate), pub(super), pub(in path)
                if let Some(end) = rest.find(')') {
                    rest[end + 1..].trim_start()
                } else {
                    byte_offset += line.len() as u32 + 1;
                    continue;
                }
            } else {
                rest
            }
        } else {
            trimmed
        };

        // Strip optional modifiers: async, const, unsafe, extern
        let mut after_mod = after_vis;
        for prefix in &["async ", "const ", "unsafe ", "extern "] {
            if let Some(rest) = after_mod.strip_prefix(prefix) {
                after_mod = rest;
            }
        }

        if let Some(rest) = after_mod.strip_prefix("fn ") {
            let rest = rest.trim_start();
            if let Some(name_end) = rest.find('(') {
                let name = rest[..name_end].trim().to_string();
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    let offset = byte_offset;
                    let length = line.len() as u32;
                    results.push((offset, length, name));
                }
            }
        }
        byte_offset += line.len() as u32 + 1;
    }
    results
}

// ── Capability handler ────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn basalt_capability_handle(
    cap_ptr: i32,
    cap_len: i32,
    req_ptr: i32,
    req_len: i32,
) -> i64 {
    let capability = unsafe {
        let bytes = core::slice::from_raw_parts(cap_ptr as *const u8, cap_len as usize);
        core::str::from_utf8_unchecked(bytes)
    };

    if capability != "semantic-facts" {
        return pack_empty();
    }

    // Request format: [src_len: u32 LE][src_bytes][path_len: u32 LE][path_bytes][root_len: u32 LE][root_bytes]
    let request = unsafe {
        core::slice::from_raw_parts(req_ptr as *const u8, req_len as usize)
    };

    if request.len() < 4 {
        return pack_empty();
    }

    let src_len = u32::from_le_bytes([request[0], request[1], request[2], request[3]]) as usize;
    if request.len() < 4 + src_len + 4 {
        return pack_empty();
    }

    let src = &request[4..4 + src_len];
    let path_len = u32::from_le_bytes([
        request[4 + src_len],
        request[5 + src_len],
        request[6 + src_len],
        request[7 + src_len],
    ]) as usize;
    let path_start = 8 + src_len;
    if request.len() < path_start + path_len {
        return pack_empty();
    }

    let path = unsafe {
        let bytes = &request[path_start..path_start + path_len];
        core::str::from_utf8_unchecked(bytes)
    };

    // Parse workspace root (optional — newer hosts include it)
    let root_start = path_start + path_len;
    let workspace_root = if request.len() >= root_start + 4 {
        let root_len = u32::from_le_bytes([
            request[root_start],
            request[root_start + 1],
            request[root_start + 2],
            request[root_start + 3],
        ]) as usize;
        if request.len() >= root_start + 4 + root_len {
            unsafe {
                let bytes = &request[root_start + 4..root_start + 4 + root_len];
                core::str::from_utf8_unchecked(bytes)
            }
        } else {
            ""
        }
    } else {
        ""
    };

    let facts = extract_semantic_facts_for_file(src, path, workspace_root);

    if facts.is_empty() {
        return pack_empty();
    }

    let serialized = serialize_facts(&facts);
    pack_output(serialized) as i64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_traversal_declarations() {
        let code = r#"
            pub struct Config {
                pub timeout: u64,
            }
            pub enum Status {
                Active,
                Inactive,
            }
            pub trait Service {
                fn run(&self);
            }
            impl Service for Config {
                fn run(&self) {}
            }
        "#;
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "test.rs", "");

        let struct_fact = facts.iter().find(|f| matches!(f, SemanticFact::DeclareSymbol { name, kind: SymbolKind::Struct, .. } if name == "Config"));
        assert!(struct_fact.is_some(), "Config struct should be declared");

        let enum_fact = facts.iter().find(|f| matches!(f, SemanticFact::DeclareSymbol { name, kind: SymbolKind::Enum, .. } if name == "Status"));
        assert!(enum_fact.is_some(), "Status enum should be declared");

        let trait_fact = facts.iter().find(|f| matches!(f, SemanticFact::DeclareSymbol { name, kind: SymbolKind::Trait, .. } if name == "Service"));
        assert!(trait_fact.is_some(), "Service trait should be declared");

        let impl_fact = facts.iter().find(|f| matches!(f, SemanticFact::Implements { type_name, contract, .. } if type_name == "Config" && contract == "Service"));
        assert!(impl_fact.is_some(), "Config implements Service should be recorded");
    }

    #[test]
    fn test_ast_traversal_control_flow_and_calls() {
        let code = r#"
            async fn process(items: Vec<i32>) {
                if items.is_empty() {
                    return;
                }
                for item in items {
                    log_item(item);
                }
                let res = fetch().await;
            }
        "#;
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "test.rs", "");

        let fn_decl = facts.iter().find(|f| matches!(f, SemanticFact::DeclareSymbol { name, kind: SymbolKind::Function, .. } if name == "process"));
        assert!(fn_decl.is_some(), "process fn should be declared");

        let has_fn_flow = facts.iter().any(|f| matches!(f, SemanticFact::ControlFlowEnter { kind: ControlFlowKind::Function, .. }));
        assert!(has_fn_flow, "Function control flow enter should be present");

        let has_cond_flow = facts.iter().any(|f| matches!(f, SemanticFact::ControlFlowEnter { kind: ControlFlowKind::Conditional, .. }));
        assert!(has_cond_flow, "Conditional control flow enter should be present");

        let has_loop_flow = facts.iter().any(|f| matches!(f, SemanticFact::ControlFlowEnter { kind: ControlFlowKind::Loop, .. }));
        assert!(has_loop_flow, "Loop control flow enter should be present");

        let has_method_call = facts.iter().any(|f| matches!(f, SemanticFact::Calls { callee, .. } if callee == "is_empty"));
        assert!(has_method_call, "is_empty call should be present");

        let has_fn_call = facts.iter().any(|f| matches!(f, SemanticFact::Calls { callee, .. } if callee == "log_item"));
        assert!(has_fn_call, "log_item call should be present");

        let has_await = facts.iter().any(|f| matches!(f, SemanticFact::AsyncBoundary { boundary_kind: AsyncBoundaryKind::Await, .. }));
        assert!(has_await, "Await async boundary should be present");
    }

    #[test]
    fn test_ast_traversal_mutations() {
        let code = r#"
            fn update(mut state: State) {
                state.count = 42;
                state.total += 10;
            }
        "#;
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "test.rs", "");

        let assign_mut = facts.iter().find(|f| matches!(f, SemanticFact::Mutation { name, .. } if name == "state.count"));
        assert!(assign_mut.is_some(), "state.count mutation should be present");

        let compound_mut = facts.iter().find(|f| matches!(f, SemanticFact::Mutation { name, .. } if name == "state.total"));
        assert!(compound_mut.is_some(), "state.total compound mutation should be present");
    }

    #[test]
    fn test_ast_traversal_io_operations() {
        let code = r#"
            async fn handle_data() {
                let text = std::fs::read_to_string("file.txt").unwrap();
                let client = reqwest::Client::new();
                let resp = fetch("https://example.com").await;
                std::fs::write("output.txt", text).unwrap();
                println!("Done");
            }
        "#;
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "test.rs", "");

        let has_read = facts.iter().any(|f| matches!(f, SemanticFact::IOOperation { io_kind: IOKind::FileRead, .. }));
        assert!(has_read, "FileRead I/O operation should be detected");

        let has_net = facts.iter().any(|f| matches!(f, SemanticFact::IOOperation { io_kind: IOKind::Network, .. }));
        assert!(has_net, "Network I/O operation should be detected");

        let has_write = facts.iter().any(|f| matches!(f, SemanticFact::IOOperation { io_kind: IOKind::FileWrite, .. }));
        assert!(has_write, "FileWrite I/O operation should be detected");

        let has_stdio = facts.iter().any(|f| matches!(f, SemanticFact::IOOperation { io_kind: IOKind::StdIO, .. }));
        assert!(has_stdio, "StdIO I/O operation should be detected");
    }

    #[test]
    fn test_syntax_error_fallback_to_regex() {
        // Incomplete / invalid Rust syntax
        let code = r#"
            use std::collections::HashMap;
            impl Foo for Bar {
            // unclosed bracket
        "#;
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "test.rs", "");

        let import_fact = facts.iter().find(|f| matches!(f, SemanticFact::ImportModule { module_path, .. } if module_path == "std::collections::HashMap"));
        assert!(import_fact.is_some(), "Regex fallback should extract import even on syntax error");

        let impl_fact = facts.iter().find(|f| matches!(f, SemanticFact::Implements { type_name, contract, .. } if type_name == "Bar" && contract == "Foo"));
        assert!(impl_fact.is_some(), "Regex fallback should extract impl even on syntax error");
    }

    #[test]
    fn test_print_real_code_output() {
        let path = "src/lib.rs";
        let src = std::fs::read(path).expect("read real source file");
        let facts = extract_semantic_facts_for_file(&src, path, "");
        println!("\n=== Extracted {} SemanticFacts from {} ===", facts.len(), path);
        for (i, fact) in facts.iter().enumerate() {
            match fact {
                SemanticFact::DeclareSymbol { kind, name, offset, length, .. } => {
                    println!("[{:02}] DeclareSymbol: {:?} '{}' @ {}..{}", i, kind, name, offset, offset + length);
                }
                SemanticFact::ControlFlowEnter { kind, offset, length } => {
                    println!("[{:02}] ControlFlowEnter: {:?} @ {}..{}", i, kind, offset, offset + length);
                }
                SemanticFact::ControlFlowExit { offset, length } => {
                    println!("[{:02}] ControlFlowExit @ {}..{}", i, offset, offset + length);
                }
                SemanticFact::Calls { callee, caller_offset, caller_length } => {
                    println!("[{:02}] Calls: '{}' @ {}..{}", i, callee, caller_offset, caller_offset + caller_length);
                }
                SemanticFact::Mutation { target, kind, name, offset, length } => {
                    println!("[{:02}] Mutation: {:?} {:?} on '{}' @ {}..{}", i, target, kind, name, offset, offset + length);
                }
                SemanticFact::AsyncBoundary { boundary_kind, offset, length } => {
                    println!("[{:02}] AsyncBoundary: {:?} @ {}..{}", i, boundary_kind, offset, offset + length);
                }
                SemanticFact::ImportModule { module_path, offset, length, .. } => {
                    println!("[{:02}] ImportModule: '{}' @ {}..{}", i, module_path, offset, offset + length);
                }
                SemanticFact::Implements { type_name, contract, offset } => {
                    println!("[{:02}] Implements: '{}' for '{}' @ {}", i, contract, type_name, offset);
                }
                other => {
                    println!("[{:02}] Other: {:?}", i, other);
                }
            }
        }
    }
}

#[cfg(test)]
mod qualified_name_tests {
    use super::*;

    #[test]
    fn declares_carry_module_qualified_names() {
        let code = "pub struct Config { pub x: i32 }\nfn helper() {}\n";
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "src/config.rs", "");
        let st = facts.iter().find_map(|f| match f {
            SemanticFact::DeclareSymbol { name, qualified_name, .. } if name == "Config" => Some(qualified_name.clone()),
            _ => None,
        }).flatten();
        assert_eq!(st, Some("config::Config".to_string()));
        let qfn = facts.iter().find_map(|f| match f {
            SemanticFact::DeclareSymbol { name, qualified_name, .. } if name == "helper" => Some(qualified_name.clone()),
            _ => None,
        }).flatten();
        assert_eq!(qfn, Some("config::helper".to_string()));
    }

    #[test]
    fn crate_roots_stay_bare() {
        let code = "fn main() {}\n";
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "src/main.rs", "");
        let q = facts.iter().find_map(|f| match f {
            SemanticFact::DeclareSymbol { name, qualified_name, .. } if name == "main" => Some(qualified_name.clone()),
            _ => None,
        }).flatten();
        assert_eq!(q, None);
    }

    #[test]
    fn methods_qualify_with_impl_type_and_self_calls_resolve() {
        let code = r#"
            pub struct Store { items: Vec<String> }
            impl Store {
                pub fn len(&self) -> usize { self.items.len() }
                pub fn is_empty(&self) -> bool { self.len() == 0 }
            }
        "#;
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "store.rs", "");
        let qm = facts.iter().find_map(|f| match f {
            SemanticFact::DeclareSymbol { name, qualified_name, .. } if name == "len" => Some(qualified_name.clone()),
            _ => None,
        }).flatten();
        assert_eq!(qm, Some("store::Store::len".to_string()));
        // Direct `self.len()` → `Store::len` for exact matching.
        assert!(
            facts.iter().any(|f| matches!(f, SemanticFact::Calls { callee, .. } if callee == "Store::len")),
            "self.len() should resolve to Store::len"
        );
        // `self.items.len()` is Vec::len — receiver type unknown, stays bare
        // so core falls back to same-file matching instead of a false exact.
        assert!(
            facts.iter().any(|f| matches!(f, SemanticFact::Calls { callee, .. } if callee == "len")),
            "self.items.len() should stay bare"
        );
    }

    #[test]
    fn nested_modules_nest_qualification() {
        let code = "mod inner {\n pub fn deep() {}\n}\n";
        let facts = extract_semantic_facts_for_file(code.as_bytes(), "outer.rs", "");
        let q = facts.iter().find_map(|f| match f {
            SemanticFact::DeclareSymbol { name, qualified_name, .. } if name == "deep" => Some(qualified_name.clone()),
            _ => None,
        }).flatten();
        assert_eq!(q, Some("outer::inner::deep".to_string()));
    }

    #[test]
    fn module_name_for_path_handles_roots() {
        assert_eq!(module_name_for_path("src/foo.rs"), Some("foo".to_string()));
        assert_eq!(module_name_for_path("src/main.rs"), None);
        assert_eq!(module_name_for_path("src/lib.rs"), None);
        assert_eq!(module_name_for_path("src/auth/mod.rs"), Some("auth".to_string()));
    }
}

// ── Review Actions & Build/Test Providers ────────────────────────────────────


#[basalt_plugin]
fn review_actions(_workspace_root: &str, _session_workspace: &str) -> Vec<ReviewActionDescriptor> {
    vec![
        ReviewActionDescriptor {
            id: "cargo-check".into(),
            title: "Cargo Check (Rust)".into(),
            kind: ReviewActionKind::Build,
            ecosystem: "rust".into(),
            command_preview: "cargo check --message-format=json".into(),
            mutates_workspace: false,
            priority: 100,
        },
        ReviewActionDescriptor {
            id: "cargo-test".into(),
            title: "Cargo Test (Rust)".into(),
            kind: ReviewActionKind::Test,
            ecosystem: "rust".into(),
            command_preview: "cargo test".into(),
            mutates_workspace: false,
            priority: 100,
        },
    ]
}

#[basalt_plugin]
fn review_action_plan(
    action_id: &str,
    _workspace_root: &str,
    _session_workspace: &str,
) -> Option<ReviewActionExecutionPlan> {
    match action_id {
        "cargo-check" => Some(ReviewActionExecutionPlan {
            executable: "cargo".into(),
            args: vec!["check".into(), "--message-format=json".into()],
            env: Vec::new(),
            cwd_mode: ReviewActionCwdMode::SessionWorkspace,
            output_category: "build".into(),
        }),
        "cargo-test" => Some(ReviewActionExecutionPlan {
            executable: "cargo".into(),
            args: vec!["test".into()],
            env: Vec::new(),
            cwd_mode: ReviewActionCwdMode::SessionWorkspace,
            output_category: "test".into(),
        }),
        _ => None,
    }
}

#[basalt_plugin]
fn review_action_parse_line(
    action_id: &str,
    line: &[u8],
    state: &[u8],
) -> (Vec<u8>, Vec<AgentEvent>) {
    let Ok(line_str) = core::str::from_utf8(line) else {
        return (state.to_vec(), Vec::new());
    };
    let trimmed = line_str.trim();
    if trimmed.is_empty() {
        return (state.to_vec(), Vec::new());
    }

    let mut events = Vec::new();

    if action_id == "cargo-check" {
        if trimmed.starts_with('{') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if v.get("reason").and_then(|r| r.as_str()) == Some("compiler-message") {
                    if let Some(msg_obj) = v.get("message") {
                        let level = msg_obj.get("level").and_then(|l| l.as_str()).unwrap_or("error");
                        let message = msg_obj.get("message").and_then(|m| m.as_str()).unwrap_or("");
                        let code = msg_obj.get("code").and_then(|c| c.get("code")).and_then(|c| c.as_str());
                        let spans = msg_obj.get("spans").and_then(|s| s.as_array());
                        let primary_span = spans.and_then(|arr| {
                            arr.iter().find(|s| s.get("is_primary").and_then(|p| p.as_bool()).unwrap_or(false))
                               .or_else(|| arr.first())
                        });

                        let file = primary_span.and_then(|s| s.get("file_name")).and_then(|f| f.as_str()).unwrap_or("");
                        let line_num = primary_span.and_then(|s| s.get("line_start")).and_then(|l| l.as_u64()).unwrap_or(1) as u32;
                        let col_num = primary_span.and_then(|s| s.get("column_start")).and_then(|c| c.as_u64()).unwrap_or(1) as u32;
                        let label = primary_span.and_then(|s| s.get("label")).and_then(|l| l.as_str());

                        if !message.is_empty() && (level == "error" || level == "warning") {
                            let diag_obj = serde_json::json!({
                                "file": file,
                                "line": line_num,
                                "col": col_num,
                                "severity": level,
                                "code": code,
                                "message": message,
                                "label": label,
                            });
                            events.push(AgentEvent::NewEntry {
                                vendor_id: format!("cargo-{}", trimmed.len()),
                                tool: "cargo-check".into(),
                                category: "build".into(),
                                raw_cmd: diag_obj.to_string(),
                                file_paths: if file.is_empty() { Vec::new() } else { vec![file.to_string()] },
                            });
                        }
                    }
                }
            }
        } else if trimmed.contains("error[E") || trimmed.starts_with("error:") {
            events.push(AgentEvent::NewEntry {
                vendor_id: format!("cargo-err-{}", trimmed.len()),
                tool: "cargo-check".into(),
                category: "build".into(),
                raw_cmd: trimmed.to_string(),
                file_paths: Vec::new(),
            });
        }
    } else if action_id == "cargo-test" {
        if trimmed.contains(" ... FAILED") {
            events.push(AgentEvent::CloseEntry {
                vendor_id: format!("test-{}", trimmed.len()),
                exit_code: 1,
                output_lines: vec![trimmed.to_string()],
            });
        } else if trimmed.contains(" ... ok") {
            events.push(AgentEvent::CloseEntry {
                vendor_id: format!("test-{}", trimmed.len()),
                exit_code: 0,
                output_lines: vec![trimmed.to_string()],
            });
        }
    }

    (state.to_vec(), events)
}

// ── WASM Probe & Verification Support (CAP_WASM_VERIFY) ────────────────────

/// Generates a `#![no_std]` harness template embedding a candidate pure function
/// and exposing the standard `basalt_probe` entrypoint.
pub fn generate_wasm_harness(symbol: &str, variant_body: &str) -> String {
    format!(
r#"#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {{
    core::arch::wasm32::unreachable()
}}

{variant_body}

#[no_mangle]
pub extern "C" fn basalt_probe(arg: u64) -> u64 {{
    {symbol}(arg)
}}
"#
    )
}

/// Command arguments for compiling the harness directly to a WASM module via rustc.
pub fn wasm_compile_plan(harness_path: &str, output_wasm_path: &str) -> (String, Vec<String>) {
    (
        "rustc".to_string(),
        vec![
            "--crate-type".to_string(),
            "cdylib".to_string(),
            "--target".to_string(),
            "wasm32-unknown-unknown".to_string(),
            "-O".to_string(),
            "-o".to_string(),
            output_wasm_path.to_string(),
            harness_path.to_string(),
        ],
    )
}

/// Default representative benchmark and verification test inputs for numeric functions.
pub fn wasm_default_inputs() -> Vec<u64> {
    vec![
        0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
        1000, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        100_000, 1_000_000, 10_000_000, 100_000_000,
        u32::MAX as u64, (u32::MAX as u64) + 1, u64::MAX / 2, u64::MAX - 1,
    ]
}

#[cfg(test)]
mod review_action_tests {
    use super::*;

    #[test]
    fn test_rust_review_actions_descriptors() {
        let actions = review_actions("/tmp/ws", "/tmp/ws/.shadow");
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].id, "cargo-check");
        assert_eq!(actions[0].kind, ReviewActionKind::Build);
        assert_eq!(actions[1].id, "cargo-test");
        assert_eq!(actions[1].kind, ReviewActionKind::Test);
    }

    #[test]
    fn test_rust_review_action_plan() {
        let plan = review_action_plan("cargo-check", "/tmp", "/tmp").unwrap();
        assert_eq!(plan.executable, "cargo");
        assert_eq!(plan.args, vec!["check", "--message-format=json"]);

        let plan_t = review_action_plan("cargo-test", "/tmp", "/tmp").unwrap();
        assert_eq!(plan_t.executable, "cargo");
        assert_eq!(plan_t.args, vec!["test"]);
    }

    #[test]
    fn test_rust_review_action_parse_compiler_json() {
        let json_line = br#"{"reason":"compiler-message","package_id":"foo","message":{"rendered":"error","code":{"code":"E0308"},"level":"error","message":"mismatched types","spans":[{"file_name":"src/main.rs","line_start":10,"column_start":5}]}}"#;
        let (_, events) = review_action_parse_line("cargo-check", json_line, &[]);
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::NewEntry { raw_cmd, category, .. } => {
                assert_eq!(category, "build");
                assert!(raw_cmd.contains("E0308"));
                assert!(raw_cmd.contains("mismatched types"));
            }
            _ => panic!("expected NewEntry with parsed JSON diagnostic"),
        }
    }

    #[test]
    fn test_wasm_harness_generation_and_plan() {
        let harness = generate_wasm_harness("add_one", "pub fn add_one(x: u64) -> u64 { x + 1 }");
        assert!(harness.contains("#![no_std]"));
        assert!(harness.contains("pub extern \"C\" fn basalt_probe"));
        assert!(harness.contains("add_one(arg)"));

        let (prog, args) = wasm_compile_plan("harness.rs", "out.wasm");
        assert_eq!(prog, "rustc");
        assert!(args.contains(&"wasm32-unknown-unknown".to_string()));
        assert!(args.contains(&"cdylib".to_string()));

        let inputs = wasm_default_inputs();
        assert_eq!(inputs.len(), 32);
    }
}


