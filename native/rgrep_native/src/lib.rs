//! C ABI for rgrep's native half.
//!
//! Two calls, both JSON over a `char*` that the caller releases with
//! `rg_free`:
//!
//!   rg_codesplit(lang, path, content, len, err) -> units of one file
//!   rg_tokenize(tokenizer, text, len, err)      -> {"ids":[...]}
//!
//! Errors are reported out-of-band: a null return with a message in `err`,
//! which is also released with `rg_free`. Every allocation crosses the
//! boundary as a C string, so Go's cgo layer needs no knowledge of Rust types.

mod split;
mod tokenize;

use std::ffi::{c_char, CStr, CString};
use std::ptr;

use serde::Serialize;

/// Schema version of the codesplit JSON. Bump it when a field changes shape.
pub const ABI_VERSION: i32 = 1;

#[derive(Serialize)]
struct UnitJson {
    kind: String,
    name: String,
    parent: String,
    signature: String,
    doc: String,
    calls: Vec<String>,
    start_line: usize,
    end_line: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Serialize)]
struct SplitJson {
    language: String,
    degraded: bool,
    truncated: bool,
    units: Vec<UnitJson>,
}

#[derive(Serialize)]
struct IdsJson {
    ids: Vec<u32>,
}

/// rg_abi_version reports the JSON schema version.
#[no_mangle]
pub extern "C" fn rg_abi_version() -> i32 {
    ABI_VERSION
}

/// rg_languages reports the language identifiers rg_codesplit accepts.
#[no_mangle]
pub extern "C" fn rg_languages() -> *mut c_char {
    match serde_json::to_string(split::LANGUAGES) {
        Ok(json) => to_c(json),
        Err(_) => ptr::null_mut(),
    }
}

/// rg_codesplit extracts the code units of one file.
///
/// # Safety
/// `lang` and `path` must be valid NUL-terminated strings, `content` must point
/// at `content_len` readable bytes (it does not need a terminator), and
/// `err_out`, when not null, must point at a writable `char*`.
#[no_mangle]
pub unsafe extern "C" fn rg_codesplit(
    lang: *const c_char,
    path: *const c_char,
    content: *const c_char,
    content_len: usize,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    let _ = path; // the extension was already mapped to `lang` by the caller
    let Some(lang) = as_str(lang) else {
        set_err(err_out, "language is not valid UTF-8");
        return ptr::null_mut();
    };
    if !split::LANGUAGES.contains(&lang) {
        // Unknown language is not a failure: the caller indexes fixed-size
        // windows instead, and `degraded` tells it why.
        let payload = SplitJson {
            language: lang.to_string(),
            degraded: true,
            truncated: false,
            units: Vec::new(),
        };
        return to_json(&payload);
    }

    let source = String::from_utf8_lossy(as_bytes(content, content_len));
    match split::split(lang, &source) {
        Ok(units) => {
            let payload = SplitJson {
                language: lang.to_string(),
                degraded: false,
                truncated: false,
                units: units
                    .into_iter()
                    .map(|u| UnitJson {
                        kind: u.kind,
                        name: u.name,
                        parent: u.parent,
                        signature: u.signature,
                        doc: u.doc,
                        calls: u.calls,
                        start_line: u.start_line,
                        end_line: u.end_line,
                        start_byte: u.start_byte,
                        end_byte: u.end_byte,
                    })
                    .collect(),
            };
            to_json(&payload)
        }
        Err(e) => {
            set_err(err_out, &e);
            ptr::null_mut()
        }
    }
}

/// rg_tokenize returns the token ids of `text`, without special tokens.
///
/// # Safety
/// `tokenizer_path` must be a valid NUL-terminated string, `text` must point at
/// `text_len` readable bytes, and `err_out` must be usable when not null.
#[no_mangle]
pub unsafe extern "C" fn rg_tokenize(
    tokenizer_path: *const c_char,
    text: *const c_char,
    text_len: usize,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    let Some(path) = as_str(tokenizer_path) else {
        set_err(err_out, "tokenizer path is not valid UTF-8");
        return ptr::null_mut();
    };
    let source = String::from_utf8_lossy(as_bytes(text, text_len));
    match tokenize::tokenize(path, &source) {
        Ok(ids) => to_json(&IdsJson { ids }),
        Err(e) => {
            set_err(err_out, &e);
            ptr::null_mut()
        }
    }
}

/// rg_free releases a string returned by this library.
///
/// # Safety
/// `ptr` must be null or a pointer previously returned by this library.
#[no_mangle]
pub unsafe extern "C" fn rg_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

unsafe fn as_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

unsafe fn as_bytes<'a>(p: *const c_char, len: usize) -> &'a [u8] {
    if p.is_null() || len == 0 {
        return &[];
    }
    std::slice::from_raw_parts(p as *const u8, len)
}

unsafe fn set_err(err_out: *mut *mut c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    *err_out = to_c(msg.to_string());
}

fn to_json<T: Serialize>(value: &T) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(json) => to_c(json),
        Err(_) => ptr::null_mut(),
    }
}

fn to_c(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_one() {
        assert_eq!(rg_abi_version(), 1);
    }

    #[test]
    fn languages_round_trip() {
        let p = rg_languages();
        assert!(!p.is_null());
        let json = unsafe { CStr::from_ptr(p).to_str().unwrap().to_string() };
        unsafe { rg_free(p) };
        let langs: Vec<String> = serde_json::from_str(&json).unwrap();
        assert!(langs.contains(&"go".to_string()));
        assert!(langs.contains(&"tsx".to_string()));
    }

    #[test]
    fn codesplit_go_source() {
        let src = "package p\n\nfunc f() { g() }\n";
        let lang = CString::new("go").unwrap();
        let path = CString::new("a/b.go").unwrap();
        let mut err: *mut c_char = ptr::null_mut();
        let out = unsafe { rg_codesplit(lang.as_ptr(), path.as_ptr(), src.as_ptr() as *const c_char, src.len(), &mut err) };
        assert!(!out.is_null(), "err: {:?}", unsafe { as_str(err) });
        let json = unsafe { CStr::from_ptr(out).to_str().unwrap().to_string() };
        unsafe { rg_free(out) };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["language"], "go");
        assert_eq!(v["degraded"], false);
        assert_eq!(v["units"][0]["name"], "f");
        assert_eq!(v["units"][0]["calls"][0], "g");
        assert_eq!(v["units"][0]["start_line"], 3);
    }

    #[test]
    fn unknown_language_is_degraded_not_failed() {
        let lang = CString::new("cobol").unwrap();
        let path = CString::new("x.cob").unwrap();
        let src = "IDENTIFICATION DIVISION.";
        let mut err: *mut c_char = ptr::null_mut();
        let out = unsafe { rg_codesplit(lang.as_ptr(), path.as_ptr(), src.as_ptr() as *const c_char, src.len(), &mut err) };
        assert!(!out.is_null());
        let json = unsafe { CStr::from_ptr(out).to_str().unwrap().to_string() };
        unsafe { rg_free(out) };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["degraded"], true);
        assert_eq!(v["units"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn tokenize_reports_a_missing_file() {
        let path = CString::new("/nonexistent/tokenizer.json").unwrap();
        let text = "x";
        let mut err: *mut c_char = ptr::null_mut();
        let out = unsafe { rg_tokenize(path.as_ptr(), text.as_ptr() as *const c_char, text.len(), &mut err) };
        assert!(out.is_null());
        assert!(!err.is_null());
        unsafe { rg_free(err) };
    }
}
