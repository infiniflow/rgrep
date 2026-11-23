//! Static (Model2Vec) embedding, as published by minishlab/model2vec.
//!
//! A Model2Vec model is not a neural network: it is a token embedding table plus
//! a mean. So "full support" means loading exactly the files the Python package
//! writes and reproducing its arithmetic:
//!
//! ```text
//! model.safetensors  embeddings [rows, dim]   f32 | f16 | f64 | i8
//!                    weights    [vocab]        optional, per-token multiplier
//!                    mapping    [vocab]        optional, token id -> table row
//! config.json        normalize (default false), max_length (default 512)
//! tokenizer.json     HF tokenizer; encoded without special tokens, unk dropped
//! ```
//!
//! Both optional tensors are supported because skipping either silently changes
//! every vector: `mapping` is what a vocabulary-quantized model stores instead of
//! a row per token, and `weights` multiplies each token's row. The Python
//! implementation is the reference; `encode_one` follows `_encode_batch`.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::fs;
use std::path::Path;
use std::ptr;

use serde::Serialize;
use serde_json::Value;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Matches `model2vec.model.DEFAULT_MAX_LENGTH`: a config without `max_length`
/// truncates at 512 tokens, exactly like `from_pretrained` does.
const DEFAULT_MAX_LENGTH: usize = 512;

/// One loaded static model.
pub struct StaticModel {
    dim: usize,
    rows: usize,
    vocab: usize,
    /// Row-major table, `rows * dim`, expanded to f32 at load time.
    table: Vec<f32>,
    /// Per-token multiplier, indexed by the *original* token id.
    weights: Option<Vec<f32>>,
    /// Token id -> table row, for vocabulary-quantized models.
    mapping: Option<Vec<u32>>,
    tokenizer: Tokenizer,
    unk_token_id: Option<u32>,
    normalize: bool,
    max_length: Option<usize>,
    dtype: &'static str,
}

#[derive(Debug, Serialize)]
pub struct InfoJson {
    dim: usize,
    rows: usize,
    vocab: usize,
    dtype: &'static str,
    normalize: bool,
    max_length: Option<usize>,
    has_weights: bool,
    has_mapping: bool,
}

impl StaticModel {
    /// Load a model directory written by `StaticModel.save_pretrained`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let config = read_json(&dir.join("config.json")).unwrap_or(Value::Null);
        let normalize = config
            .get("normalize")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // A missing key means the documented default; `null` means "no
        // truncation". They are different, so the match has three arms.
        let max_length = match config.get("max_length") {
            None => Some(DEFAULT_MAX_LENGTH),
            Some(Value::Null) => None,
            Some(value) => value
                .as_u64()
                .map(|n| Some(n as usize))
                .ok_or_else(|| "config.json: max_length is not an integer or null".to_string())?,
        };

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer_bytes = fs::read(&tokenizer_path)
            .map_err(|e| format!("read {}: {e}", tokenizer_path.display()))?;
        let tokenizer_json: Value = serde_json::from_slice(&tokenizer_bytes)
            .map_err(|e| format!("parse {}: {e}", tokenizer_path.display()))?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| format!("load {}: {e}", tokenizer_path.display()))?;
        // The tokenizer's own truncation (often 512) is a property of the teacher
        // model, not of this index. Ours comes from config.max_length so the
        // query side and the document side cannot disagree.
        tokenizer
            .with_truncation(max_length.map(|max_length| TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| format!("configure tokenizer truncation: {e}"))?;
        let unk_token_id = tokenizer_json
            .get("model")
            .and_then(|model| model.get("unk_token"))
            .and_then(Value::as_str)
            .and_then(|unk| tokenizer.token_to_id(unk));

        let safetensors_path = dir.join("model.safetensors");
        let bytes = fs::read(&safetensors_path)
            .map_err(|e| format!("read {}: {e}", safetensors_path.display()))?;
        let file = Safetensors::parse(&bytes)?;

        let embeddings = file.take(&bytes, "embeddings")?;
        let shape: Vec<usize> = embeddings.shape().iter().map(|d| *d as usize).collect();
        if shape.len() != 2 {
            return Err("model.safetensors: embeddings is not two-dimensional".to_string());
        }
        let (rows, dim) = (shape[0], shape[1]);
        if rows == 0 || dim == 0 {
            return Err("model.safetensors: embeddings is empty".to_string());
        }
        let table = embeddings.as_f32("embeddings")?;
        if table.len() != rows * dim {
            return Err(format!(
                "model.safetensors: embeddings declares {rows}x{dim} but holds {} values",
                table.len()
            ));
        }

        let mapping = match file.get("mapping") {
            Some(_) => Some(file.take(&bytes, "mapping")?.as_u32("mapping")?),
            None => None,
        };
        let weights = match file.get("weights") {
            Some(_) => Some(file.take(&bytes, "weights")?.as_f32("weights")?),
            None => None,
        };

        // Upstream indexes weights and mapping by token id, so a model whose
        // vocabulary is larger than its table must supply `mapping`; without it
        // any token above `rows` would fail per document instead of at load.
        let vocab = mapping.as_ref().map_or(rows, Vec::len);
        if let Some(weights) = &weights {
            if weights.len() < vocab {
                return Err(format!(
                    "model.safetensors: weights has {} entries but the vocabulary needs {vocab}",
                    weights.len()
                ));
            }
        }
        if let Some(mapping) = &mapping {
            if let Some(bad) = mapping.iter().position(|row| *row as usize >= rows) {
                return Err(format!(
                    "model.safetensors: mapping[{bad}] = {} is outside the {rows}-row table",
                    mapping[bad]
                ));
            }
        }

        Ok(Self {
            dim,
            rows,
            vocab,
            table,
            weights,
            mapping,
            tokenizer,
            unk_token_id,
            normalize,
            max_length,
            dtype: embeddings.dtype_label(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn info(&self) -> InfoJson {
        InfoJson {
            dim: self.dim,
            rows: self.rows,
            vocab: self.vocab,
            dtype: self.dtype,
            normalize: self.normalize,
            max_length: self.max_length,
            has_weights: self.weights.is_some(),
            has_mapping: self.mapping.is_some(),
        }
    }

    /// Encode `texts` into `out` (row-major, `texts.len() * dim` floats).
    pub fn encode_into(&self, texts: &[&str], out: &mut [f32]) -> Result<(), String> {
        if out.len() < texts.len() * self.dim {
            return Err(format!(
                "output buffer holds {} floats, {} needed",
                out.len(),
                texts.len() * self.dim
            ));
        }
        for (index, text) in texts.iter().enumerate() {
            let vector = self.encode_one(text)?;
            let start = index * self.dim;
            out[start..start + self.dim].copy_from_slice(&vector);
        }
        Ok(())
    }

    /// One text: tokenize, drop unk, map, weight, mean, normalize.
    pub fn encode_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if let Some(unk) = self.unk_token_id {
            // Upstream drops unknown tokens instead of averaging in a meaningless
            // [UNK] row; word-level models depend on this.
            ids.retain(|id| *id != unk);
        }
        if ids.is_empty() {
            // `_encode_batch` leaves an all-zero row for an empty token list.
            return Ok(vec![0.0; self.dim]);
        }

        // f64 accumulation: the reference implementation sums in the table's
        // dtype, but cosine differences from f64 are ~1e-8 while f32 drift over
        // hundreds of tokens is measurable. Do not "optimise" this into mul_add.
        let mut sum = vec![0.0_f64; self.dim];
        for id in &ids {
            let row = match &self.mapping {
                Some(mapping) => *mapping
                    .get(*id as usize)
                    .ok_or_else(|| format!("token id {id} is outside the mapping table"))?
                    as usize,
                None => *id as usize,
            };
            if row >= self.rows {
                return Err(format!(
                    "token id {id} maps to row {row}, outside the {}-row table",
                    self.rows
                ));
            }
            let weight = match &self.weights {
                Some(weights) => f64::from(
                    *weights
                        .get(*id as usize)
                        .ok_or_else(|| format!("token id {id} is outside the weights table"))?,
                ),
                None => 1.0,
            };
            let start = row * self.dim;
            for (slot, value) in sum.iter_mut().zip(&self.table[start..start + self.dim]) {
                *slot += f64::from(*value) * weight;
            }
        }

        let count = ids.len() as f64;
        let mut out = vec![0.0_f32; self.dim];
        let mut squared_norm = 0.0_f64;
        for (slot, value) in out.iter_mut().zip(&sum) {
            let mean = *value / count;
            squared_norm += mean * mean;
            *slot = mean as f32;
        }
        if self.normalize && squared_norm > 0.0 {
            let inverse = squared_norm.sqrt().recip();
            for slot in &mut out {
                *slot = (f64::from(*slot) * inverse) as f32;
            }
        }
        Ok(out)
    }
}

// -- safetensors ------------------------------------------------------------

/// Header entry of one tensor; the payload stays in the file buffer.
struct TensorMeta {
    dtype: String,
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

struct Safetensors {
    data_start: usize,
    metas: HashMap<String, TensorMeta>,
}

struct TensorView<'a> {
    meta: &'a TensorMeta,
    data: &'a [u8],
}

impl Safetensors {
    /// Parse `u64` header length + JSON header. The payload is not copied.
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("model.safetensors: file is shorter than its header".to_string());
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .filter(|start| *start <= bytes.len())
            .ok_or_else(|| "model.safetensors: header length is out of range".to_string())?;
        let header: Value = serde_json::from_slice(&bytes[8..data_start])
            .map_err(|e| format!("model.safetensors: header is not JSON: {e}"))?;
        let object = header
            .as_object()
            .ok_or_else(|| "model.safetensors: header is not an object".to_string())?;

        let mut metas = HashMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let dtype = value
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("model.safetensors: {name} has no dtype"))?;
            let offsets = value
                .get("data_offsets")
                .and_then(Value::as_array)
                .filter(|offsets| offsets.len() == 2)
                .ok_or_else(|| format!("model.safetensors: {name} has no data_offsets"))?;
            let shape = value
                .get("shape")
                .and_then(Value::as_array)
                .map(|dims| dims.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            metas.insert(
                name.clone(),
                TensorMeta {
                    dtype: dtype.to_string(),
                    shape,
                    start: offsets[0].as_u64().unwrap_or(0) as usize,
                    end: offsets[1].as_u64().unwrap_or(0) as usize,
                },
            );
        }
        Ok(Self { data_start, metas })
    }

    fn get(&self, name: &str) -> Option<&TensorMeta> {
        self.metas.get(name)
    }

    fn take<'a>(&'a self, bytes: &'a [u8], name: &str) -> Result<TensorView<'a>, String> {
        let meta = self
            .metas
            .get(name)
            .ok_or_else(|| format!("model.safetensors: missing tensor {name}"))?;
        let data = self
            .data_start
            .checked_add(meta.start)
            .zip(self.data_start.checked_add(meta.end))
            .and_then(|(start, end)| bytes.get(start..end))
            .ok_or_else(|| format!("model.safetensors: {name} points outside the file"))?;
        Ok(TensorView { meta, data })
    }
}

impl TensorView<'_> {
    fn shape(&self) -> &[u64] {
        &self.meta.shape
    }

    fn dtype_label(&self) -> &'static str {
        match self.meta.dtype.as_str() {
            "F16" => "float16",
            "F32" => "float32",
            "F64" => "float64",
            "I8" => "int8",
            "I32" | "I64" => "int",
            _ => "unknown",
        }
    }

    fn as_u32(&self, name: &str) -> Result<Vec<u32>, String> {
        let view = self;
        match view.meta.dtype.as_str() {
            "I32" => view
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| {
                    let value = i32::from_le_bytes(*chunk);
                    u32::try_from(value)
                        .map_err(|_| format!("{name} holds the negative row index {value}"))
                })
                .collect(),
            "I64" => view
                .data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| {
                    let value = i64::from_le_bytes(*chunk);
                    u32::try_from(value)
                        .map_err(|_| format!("{name} holds the negative row index {value}"))
                })
                .collect(),
            other => Err(format!(
                "{name} has unsupported dtype {other}; expected I32 or I64"
            )),
        }
    }

    fn as_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        match self.meta.dtype.as_str() {
            "F32" => Ok(self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()),
            "F16" => Ok(self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
                .collect()),
            "F64" => Ok(self
                .data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| f64::from_le_bytes(*chunk) as f32)
                .collect()),
            "I8" => Ok(self
                .data
                .iter()
                .map(|byte| f32::from(i8::from_ne_bytes([*byte])))
                .collect()),
            other => Err(format!(
                "{name} has unsupported dtype {other}; expected F32, F16, F64 or I8"
            )),
        }
    }
}

/// `f16 -> f32` without pulling in the `half` crate for one conversion.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: value * 2^-24.
        let value = (mantissa as f32) * 2.0_f32.powi(-24);
        return if sign != 0 { -value } else { value };
    }
    let out = if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

fn read_json(path: &Path) -> Option<Value> {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

// -- C ABI ------------------------------------------------------------------

/// Opaque handle returned by `rg_m2v_load`.
pub struct ModelHandle(StaticModel);

/// rg_m2v_load opens a Model2Vec model directory.
///
/// # Safety
/// `dir` must be a valid NUL-terminated string and `err_out`, when not null,
/// must point at a writable `char*`.
#[no_mangle]
pub unsafe extern "C" fn rg_m2v_load(
    dir: *const c_char,
    err_out: *mut *mut c_char,
) -> *mut ModelHandle {
    let Some(dir) = cstr(dir) else {
        set_err(err_out, "model directory is not valid UTF-8");
        return ptr::null_mut();
    };
    match StaticModel::load(Path::new(dir)) {
        Ok(model) => Box::into_raw(Box::new(ModelHandle(model))),
        Err(error) => {
            set_err(err_out, &error);
            ptr::null_mut()
        }
    }
}

/// rg_m2v_dim reports the embedding width.
///
/// # Safety
/// `handle` must be null or a pointer returned by `rg_m2v_load`.
#[no_mangle]
pub unsafe extern "C" fn rg_m2v_dim(handle: *const ModelHandle) -> usize {
    match handle.as_ref() {
        Some(handle) => handle.0.dim(),
        None => 0,
    }
}

/// rg_m2v_info reports a JSON summary of the loaded model.
///
/// # Safety
/// `handle` must be null or a pointer returned by `rg_m2v_load`.
#[no_mangle]
pub unsafe extern "C" fn rg_m2v_info(handle: *const ModelHandle) -> *mut c_char {
    let Some(handle) = handle.as_ref() else {
        return ptr::null_mut();
    };
    match serde_json::to_string(&handle.0.info()) {
        Ok(json) => to_c(json),
        Err(_) => ptr::null_mut(),
    }
}

/// rg_m2v_encode writes `count * dim` floats into `out`; returns 0 on success.
///
/// # Safety
/// `texts` must point at `count` NUL-terminated strings, `lens` must point at
/// `count` byte lengths (excluding the terminator) or be null, `out` must point
/// at `count * dim` writable floats, and `err_out` must be usable when not null.
#[no_mangle]
pub unsafe extern "C" fn rg_m2v_encode(
    handle: *const ModelHandle,
    texts: *const *const c_char,
    lens: *const usize,
    count: usize,
    out: *mut f32,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(handle) = handle.as_ref() else {
        set_err(err_out, "model handle is null");
        return -1;
    };
    if count == 0 {
        return 0;
    }
    if texts.is_null() || out.is_null() {
        set_err(err_out, "texts or output buffer is null");
        return -1;
    }
    let pointers = std::slice::from_raw_parts(texts, count);
    let lengths = if lens.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(lens, count))
    };
    let mut owned = Vec::with_capacity(count);
    for (index, pointer) in pointers.iter().enumerate() {
        if pointer.is_null() {
            set_err(err_out, "a text pointer is null");
            return -1;
        }
        match lengths {
            Some(lengths) => {
                let bytes = std::slice::from_raw_parts(*pointer as *const u8, lengths[index]);
                owned.push(String::from_utf8_lossy(bytes).into_owned());
            }
            None => match cstr(*pointer) {
                Some(text) => owned.push(text.to_string()),
                None => {
                    set_err(err_out, "a text is not valid UTF-8");
                    return -1;
                }
            },
        }
    }
    let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
    let dim = handle.0.dim();
    let buffer = std::slice::from_raw_parts_mut(out, count * dim);
    match handle.0.encode_into(&borrowed, buffer) {
        Ok(()) => 0,
        Err(error) => {
            set_err(err_out, &error);
            -1
        }
    }
}

/// rg_m2v_free releases a handle returned by `rg_m2v_load`.
///
/// # Safety
/// `handle` must be null or a pointer from `rg_m2v_load`, not yet freed.
#[no_mangle]
pub unsafe extern "C" fn rg_m2v_free(handle: *mut ModelHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

unsafe fn set_err(err_out: *mut *mut c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    *err_out = to_c(msg.to_string());
}

fn to_c(value: String) -> *mut c_char {
    match std::ffi::CString::new(value) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests;
