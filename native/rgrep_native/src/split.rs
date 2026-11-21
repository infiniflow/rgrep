//! Code unit extraction with tree-sitter.
//!
//! One pass over the CST produces the units the Go indexer stores: a unit is
//! the declaration's whole span, plus the signature (the text up to its body),
//! the doc comment that belongs to it, and the call names inside it. The walk
//! carries the enclosing declaration's name down as `parent`, which is how a
//! method ends up knowing its class.

use tree_sitter::{Language, Node, Parser};

/// The extraction result for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub kind: String,
    pub name: String,
    pub parent: String,
    pub signature: String,
    pub doc: String,
    pub calls: Vec<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Go,
    Python,
    Rust,
    Js,
}

const MAX_SIGNATURE: usize = 240;
const MAX_DOC: usize = 600;
const MAX_CALLS: usize = 40;

/// split parses `source` with the grammar for `lang` and returns its units.
/// An unsupported language is an error; an empty file is not.
pub fn split(lang: &str, source: &str) -> Result<Vec<Unit>, String> {
    let (language, family): (Language, Family) = match lang {
        "go" => (tree_sitter_go::LANGUAGE.into(), Family::Go),
        "python" => (tree_sitter_python::LANGUAGE.into(), Family::Python),
        "rust" => (tree_sitter_rust::LANGUAGE.into(), Family::Rust),
        "javascript" => (tree_sitter_javascript::LANGUAGE.into(), Family::Js),
        "typescript" => (tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(), Family::Js),
        "tsx" => (tree_sitter_typescript::LANGUAGE_TSX.into(), Family::Js),
        other => return Err(format!("unsupported language: {other}")),
    };

    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| format!("load grammar for {lang}: {e}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| format!("parse failed for {lang}"))?;

    let mut units = Vec::new();
    walk(
        tree.root_node(),
        None,
        false,
        family,
        source.as_bytes(),
        &mut units,
    );
    Ok(units)
}

/// Languages this crate can split, in the order `rg_languages` reports them.
pub const LANGUAGES: &[&str] = &["go", "python", "rust", "javascript", "typescript", "tsx"];

fn walk(
    node: Node,
    parent: Option<&str>,
    in_type: bool,
    family: Family,
    src: &[u8],
    out: &mut Vec<Unit>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }
        if let Some(kind) = unit_kind(family, child, in_type) {
            let name = unit_name(family, child, src);
            let unit = build_unit(child, kind, &name, parent, family, src);
            let has_name = !name.is_empty();
            out.push(unit);
            let nested_parent = if has_name { Some(name.as_str()) } else { parent };
            walk(child, nested_parent, true, family, src, out);
            continue;
        }
        // A container that is not itself a unit (a class, an impl block, a
        // module) still names the units inside it.
        if is_container(family, child) {
            let name = unit_name(family, child, src);
            let nested_parent = if name.is_empty() {
                parent
            } else {
                Some(name.as_str())
            };
            walk(child, nested_parent, true, family, src, out);
            continue;
        }
        walk(child, parent, in_type, family, src, out);
    }
}

/// unit_kind maps a CST node kind to the label rgrep stores, or None when the
/// node is not a unit. `in_type` is true inside a class / impl block, which is
/// what makes a function a method there.
fn unit_kind(family: Family, node: Node, in_type: bool) -> Option<&'static str> {
    let kind = node.kind();
    match family {
        Family::Go => match kind {
            "function_declaration" => Some("function"),
            "method_declaration" => Some("method"),
            "const_declaration" => Some("const"),
            "var_declaration" => Some("var"),
            "type_declaration" => {
                let spec = type_spec_of(node)?;
                match type_child_kind(spec) {
                    Some("struct_type") => Some("struct"),
                    Some("interface_type") => Some("interface"),
                    _ => Some("type"),
                }
            }
            _ => None,
        },
        Family::Python => match kind {
            "function_definition" => Some(if in_type { "method" } else { "function" }),
            "class_definition" => Some("class"),
            _ => None,
        },
        Family::Rust => match kind {
            "function_item" => Some(if in_type { "method" } else { "function" }),
            "struct_item" => Some("struct"),
            "enum_item" => Some("enum"),
            "trait_item" => Some("trait"),
            "impl_item" => Some("impl"),
            "const_item" => Some("const"),
            "static_item" => Some("static"),
            "type_item" => Some("type"),
            _ => None,
        },
        Family::Js => match kind {
            "function_declaration" => Some("function"),
            "generator_function_declaration" => Some("function"),
            "function_signature" => Some("function"),
            "class_declaration" => Some("class"),
            "method_definition" => Some("method"),
            "interface_declaration" => Some("interface"),
            "enum_declaration" => Some("enum"),
            "type_alias_declaration" => Some("type"),
            "variable_declarator" => match node.child_by_field_name("value").map(|v| v.kind()) {
                Some("arrow_function") | Some("function_expression") | Some("function") => {
                    Some("function")
                }
                _ => None,
            },
            _ => None,
        },
    }
}

fn is_container(family: Family, node: Node) -> bool {
    match family {
        Family::Python => node.kind() == "class_definition",
        Family::Rust => matches!(
            node.kind(),
            "impl_item" | "trait_item" | "mod_item" | "struct_item" | "enum_item"
        ),
        Family::Js => matches!(node.kind(), "class_declaration" | "class_body"),
        Family::Go => matches!(node.kind(), "type_declaration"),
    }
}

fn unit_name(family: Family, node: Node, src: &[u8]) -> String {
    let text = |n: Option<Node>| -> String {
        n.and_then(|n| n.utf8_text(src).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    match family {
        Family::Go => match node.kind() {
            "type_declaration" => type_spec_of(node)
                .map(|spec| text(spec.child_by_field_name("name")))
                .unwrap_or_default(),
            "const_declaration" | "var_declaration" => {
                first_spec_name(node, src).unwrap_or_default()
            }
            _ => text(node.child_by_field_name("name")),
        },
        Family::Python | Family::Js => text(node.child_by_field_name("name")),
        Family::Rust => {
            if node.kind() == "impl_item" {
                let ty = text(node.child_by_field_name("type"));
                simplify_type(&ty)
            } else {
                text(node.child_by_field_name("name"))
            }
        }
    }
}

fn type_spec_of(node: Node) -> Option<Node> {
    let mut cursor = node.walk();
    let found = node
        .children(&mut cursor)
        .find(|c| matches!(c.kind(), "type_spec" | "type_alias"));
    found
}

fn type_child_kind(spec: Node) -> Option<&'static str> {
    let ty = spec.child_by_field_name("type")?;
    // The grammar hands back a &'static str for its own kinds.
    Some(match ty.kind() {
        "struct_type" => "struct_type",
        "interface_type" => "interface_type",
        _ => "other",
    })
}

fn first_spec_name(node: Node, src: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    let spec = node
        .children(&mut cursor)
        .find(|c| matches!(c.kind(), "const_spec" | "var_spec"))?;
    spec.child_by_field_name("name")
        .and_then(|n| n.utf8_text(src).ok())
        .map(|s| s.trim().to_string())
}

/// go_receiver_type reads the type of a method receiver: `(s *Server)` ->
/// `Server`, `(s Server[T])` -> `Server`.
fn go_receiver_type(node: Node, src: &[u8]) -> String {
    let Some(receiver) = node.child_by_field_name("receiver") else {
        return String::new();
    };
    let mut cursor = receiver.walk();
    let param = receiver
        .named_children(&mut cursor)
        .find(|c| c.kind() == "parameter_declaration");
    let Some(param) = param else {
        return String::new();
    };
    let Some(ty) = param.child_by_field_name("type") else {
        return String::new();
    };
    let raw = ty.utf8_text(src).unwrap_or("");
    simplify_type(raw)
}

/// simplify_type trims generics and references off an impl target, so
/// `impl<'a> Foo<'a>` is stored as `Foo`.
fn simplify_type(text: &str) -> String {
    let t = text.trim();
    let t = t.split('<').next().unwrap_or(t);
    let t = t.rsplit("::").next().unwrap_or(t);
    t.trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
        .to_string()
}

fn build_unit(
    node: Node,
    kind: &'static str,
    name: &str,
    parent: Option<&str>,
    family: Family,
    src: &[u8],
) -> Unit {
    let start = node.start_byte();
    let end = node.end_byte();
    // A Go method is declared next to its type, not inside it, so the receiver
    // is the only place its parent can come from.
    let parent_name = match (family, node.kind()) {
        (Family::Go, "method_declaration") => {
            let recv = go_receiver_type(node, src);
            if recv.is_empty() {
                parent.unwrap_or("").to_string()
            } else {
                recv
            }
        }
        _ => parent.unwrap_or("").to_string(),
    };
    Unit {
        kind: kind.to_string(),
        name: name.to_string(),
        parent: parent_name,
        signature: signature(node, family, src),
        doc: doc_comment(node, family, src),
        calls: collect_calls(node, family, src),
        start_line: node.start_position().row + 1,
        end_line: node.end_position().row + 1,
        start_byte: start,
        end_byte: end,
    }
}

/// signature is the declaration text up to its body: `func f(a int) error`,
/// `class Foo(Base):`, `fn f() -> u32`. Collapsing whitespace keeps a long
/// parameter list on one line in the embedded text.
fn signature(node: Node, family: Family, src: &[u8]) -> String {
    let body = body_node(node, family);
    let end = body.map(|b| b.start_byte()).unwrap_or_else(|| node.end_byte());
    let start = node.start_byte();
    if end <= start || end > src.len() {
        return String::new();
    }
    let raw = String::from_utf8_lossy(&src[start..end]);
    let collapsed = collapse(&raw);
    truncate_chars(&collapsed, MAX_SIGNATURE)
}

fn body_node<'a>(node: Node<'a>, family: Family) -> Option<Node<'a>> {
    if let Some(body) = node.child_by_field_name("body") {
        return Some(body);
    }
    match family {
        Family::Go if node.kind() == "type_declaration" => {
            let spec = type_spec_of(node)?;
            spec.child_by_field_name("type")
        }
        Family::Js if node.kind() == "variable_declarator" => {
            let value = node.child_by_field_name("value")?;
            value.child_by_field_name("body")
        }
        _ => None,
    }
}

/// doc_comment collects the comment block directly above the declaration. A
/// comment that starts on the declaration's own line is a trailing comment of
/// the previous statement and is skipped.
fn doc_comment(node: Node, family: Family, src: &[u8]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut cur = node.prev_named_sibling();
    while let Some(prev) = cur {
        if !is_comment(prev.kind()) {
            break;
        }
        if prev.end_position().row == node.start_position().row {
            break;
        }
        let text = prev.utf8_text(src).unwrap_or("").trim().to_string();
        if !text.is_empty() {
            parts.push(text);
        }
        cur = prev.prev_named_sibling();
    }
    parts.reverse();
    let mut doc = parts.join("\n");

    if family == Family::Python {
        if let Some(ds) = python_docstring(node, src) {
            if doc.is_empty() {
                doc = ds;
            } else {
                doc = format!("{doc}\n{ds}");
            }
        }
    }
    truncate_chars(&doc, MAX_DOC)
}

fn is_comment(kind: &str) -> bool {
    matches!(kind, "comment" | "line_comment" | "block_comment" | "doc_comment")
}

fn python_docstring(node: Node, src: &[u8]) -> Option<String> {
    let body = node.child_by_field_name("body")?;
    let mut cursor = body.walk();
    let first = body.named_children(&mut cursor).next()?;
    if first.kind() != "expression_statement" {
        return None;
    }
    let mut c2 = first.walk();
    let str_node = first.named_children(&mut c2).next()?;
    if !matches!(
        str_node.kind(),
        "string" | "concatenated_string" | "string_content"
    ) {
        return None;
    }
    let raw = str_node.utf8_text(src).ok()?.trim();
    Some(
        raw.trim_matches(|c| c == '"' || c == '\'')
            .trim()
            .to_string(),
    )
}

/// collect_calls lists the function names invoked inside the body, which is a
/// strong signal for "what does this code do" queries. The walk is in source
/// order so the embedded text is stable across runs.
fn collect_calls(node: Node, family: Family, src: &[u8]) -> Vec<String> {
    let call_kind = match family {
        Family::Python => "call",
        _ => "call_expression",
    };
    let mut out: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let body = body_node(node, family).unwrap_or(node);
    collect_calls_in(body, call_kind, src, &mut seen, &mut out);
    out
}

fn collect_calls_in(
    node: Node,
    call_kind: &str,
    src: &[u8],
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<String>,
) {
    if out.len() >= MAX_CALLS {
        return;
    }
    if node.kind() == call_kind {
        if let Some(f) = node.child_by_field_name("function") {
            if let Ok(text) = f.utf8_text(src) {
                let name = last_segment(text);
                if !name.is_empty() && seen.insert(name.clone()) {
                    out.push(name);
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.is_named() {
            collect_calls_in(child, call_kind, src, seen, out);
        }
    }
}

/// last_segment reduces a call target to a bare name: `pkg.Do` -> `Do`,
/// `self.read::<T>` -> `read`, `arr[i].f` -> `f`.
fn last_segment(text: &str) -> String {
    let head = text
        .split(|c: char| c == '(' || c == '[' || c == '<' || c == '!')
        .next()
        .unwrap_or(text);
    let seg = head
        .split(|c: char| c == '.' || c == ':' || c == '&' || c == '*' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .next_back()
        .unwrap_or("");
    seg.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn collapse(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(ch);
    }
    out
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_function_and_method() {
        let src = r#"package main

// Server serves requests.
type Server struct {
	addr string
}

// Start binds and serves.
func (s *Server) Start(port int) error {
	return listen(port)
}

func helper() {
	Start(1)
}
"#;
        let units = split("go", src).unwrap();
        let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
        assert!(names.contains(&"Server"), "units: {names:?}");
        assert!(names.contains(&"Start"), "units: {names:?}");
        assert!(names.contains(&"helper"), "units: {names:?}");

        let start = units.iter().find(|u| u.name == "Start").unwrap();
        assert_eq!(start.kind, "method");
        assert_eq!(start.parent, "Server");
        assert!(start.signature.starts_with("func (s *Server) Start"));
        assert!(start.calls.iter().any(|c| c == "listen"), "calls: {:?}", start.calls);

        let server = units.iter().find(|u| u.name == "Server").unwrap();
        assert_eq!(server.kind, "struct");
        assert!(server.doc.contains("serves requests"));
    }

    #[test]
    fn python_class_method_and_docstring() {
        let src = r#"# A client.
class Client(Base):
    """HTTP client."""

    def get(self, url):
        return self._send(url)

    def _send(self, url):
        return request(url)
"#;
        let units = split("python", src).unwrap();
        let cls = units.iter().find(|u| u.name == "Client").unwrap();
        assert_eq!(cls.kind, "class");
        assert!(cls.doc.contains("HTTP client"), "doc: {:?}", cls.doc);

        let get = units.iter().find(|u| u.name == "get").unwrap();
        assert_eq!(get.parent, "Client");
        assert!(get.calls.iter().any(|c| c == "_send"), "calls: {:?}", get.calls);
    }

    #[test]
    fn rust_impl_and_trait() {
        let src = r#"pub struct Store {
    n: usize,
}

impl<'a> Store {
    pub fn get(&self, k: &str) -> Option<u8> {
        self.lookup(k)
    }
}

pub trait Reader {
    fn read(&self) -> u8;
}
"#;
        let units = split("rust", src).unwrap();
        let imp = units.iter().find(|u| u.kind == "impl").unwrap();
        assert_eq!(imp.name, "Store");
        let get = units.iter().find(|u| u.name == "get").unwrap();
        assert_eq!(get.parent, "Store");
        assert!(get.signature.contains("fn get"));
        assert!(units.iter().any(|u| u.kind == "trait" && u.name == "Reader"));
    }

    #[test]
    fn javascript_arrow_and_class() {
        let src = r#"// Fetch one user.
export const fetchUser = async (id) => {
  return api.get(id);
};

class Repo {
  save(row) {
    store.put(row);
  }
}
"#;
        let units = split("javascript", src).unwrap();
        let fetch = units.iter().find(|u| u.name == "fetchUser").unwrap();
        assert_eq!(fetch.kind, "function");
        assert!(fetch.calls.iter().any(|c| c == "get"), "calls: {:?}", fetch.calls);

        let save = units.iter().find(|u| u.name == "save").unwrap();
        assert_eq!(save.kind, "method");
        assert_eq!(save.parent, "Repo");
    }

    #[test]
    fn typescript_interface_and_tsx() {
        let src = r#"interface Point { x: number; y: number }
function dist(a: Point, b: Point): number { return Math.sqrt(len(a, b)); }
"#;
        let units = split("typescript", src).unwrap();
        assert!(units.iter().any(|u| u.kind == "interface" && u.name == "Point"));
        let dist = units.iter().find(|u| u.name == "dist").unwrap();
        assert!(dist.calls.iter().any(|c| c == "len"), "calls: {:?}", dist.calls);

        let tsx = r#"export function App(): JSX.Element {
  return <div onClick={() => go()}>{rows.map(render)}</div>;
}
"#;
        let units = split("tsx", tsx).unwrap();
        let app = units.iter().find(|u| u.name == "App").unwrap();
        assert!(app.calls.iter().any(|c| c == "go"), "calls: {:?}", app.calls);
    }

    #[test]
    fn unknown_language_is_an_error() {
        assert!(split("cobol", "x").is_err());
    }

    #[test]
    fn byte_spans_cover_the_source() {
        let src = "package p\n\nfunc a() {}\n";
        let units = split("go", src).unwrap();
        let a = &units[0];
        assert_eq!(&src[a.start_byte..a.end_byte], "func a() {}");
        assert_eq!(a.start_line, 3);
    }
}
