//! Unit tests for the static-embedding loader.
//!
//! Every model here is written from scratch (`safetensors` + `tokenizer.json` +
//! `config.json`) so the tests run without a download and pin the reading side
//! of the format: dtypes, weights, mapping, truncation and unk handling.

use super::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A tiny WordPiece vocabulary: `[UNK]` plus four words, one letter each.
const TOKENIZER: &str = r###"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}
  }
}"###;

struct ModelDir {
    path: PathBuf,
}

impl Drop for ModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// `tensors` maps a name to (dtype, shape, little-endian payload).
fn write_model(
    config: &str,
    tensors: &[(&str, &str, Vec<u64>, Vec<u8>)],
    tokenizer: &str,
) -> ModelDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "rgrep-m2v-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();

    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, data) in tensors {
        let start = payload.len();
        payload.extend_from_slice(data);
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".to_string(), (*dtype).into());
        entry.insert(
            "shape".to_string(),
            shape
                .iter()
                .map(|d| serde_json::Value::from(*d))
                .collect::<Vec<_>>()
                .into(),
        );
        entry.insert(
            "data_offsets".to_string(),
            vec![
                serde_json::Value::from(start as u64),
                serde_json::Value::from(payload.len() as u64),
            ]
            .into(),
        );
        header.insert((*name).to_string(), entry.into());
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);

    fs::write(path.join("model.safetensors"), file).unwrap();
    fs::write(path.join("config.json"), config).unwrap();
    fs::write(path.join("tokenizer.json"), tokenizer).unwrap();
    ModelDir { path }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| {
            let bits = f32_to_f16(*v);
            bits.to_le_bytes()
        })
        .collect()
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits & 0x7f_ffff) >> 13;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | mantissa as u16
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "component {index}: {a} vs {e} ({:?})",
            actual
        );
    }
}

fn mean(rows: &[[f32; 2]], ids: &[usize], normalize: bool) -> Vec<f32> {
    let mut out = [0.0_f64; 2];
    for id in ids {
        for (slot, value) in out.iter_mut().zip(rows[*id]) {
            *slot += f64::from(value);
        }
    }
    let count = ids.len() as f64;
    let mut out: Vec<f32> = out.iter().map(|v| (v / count) as f32).collect();
    if normalize {
        let norm = out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        for value in &mut out {
            *value = (*value as f64 / norm) as f32;
        }
    }
    out
}

#[test]
fn f32_table_is_averaged_and_normalized() {
    let rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let model = write_model(
        r#"{"normalize": true}"#,
        &[("embeddings", "F32", vec![5, 2], f32_bytes(&flat))],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    // "a b" -> [1, 2]; mean is [0.5, 0.5], normalized [0.7071, 0.7071].
    let vector = loaded.encode_one("a b").unwrap();
    assert_close(&vector, &mean(&rows, &[1, 2], true));
}

#[test]
fn f16_table_matches_f32_math() {
    let rows = [[0.5, -0.25], [0.125, 0.75], [0.0, 0.5], [0.0, 0.0], [1.0, 0.5]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let model = write_model(
        r#"{"normalize": false}"#,
        &[("embeddings", "F16", vec![5, 2], f16_bytes(&flat))],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    let vector = loaded.encode_one("a b c").unwrap();
    assert_close(&vector, &mean(&rows, &[1, 2, 3], false));
    assert_eq!(loaded.info().dtype, "float16");
}

#[test]
fn unknown_tokens_are_dropped() {
    let rows = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let model = write_model(
        r#"{"normalize": false}"#,
        &[("embeddings", "F32", vec![5, 2], f32_bytes(&flat))],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    // "a zzz" tokenizes to [1, UNK]; the UNK row must not be averaged in.
    assert_close(&loaded.encode_one("a zzz").unwrap(), &[0.0, 1.0]);
    // All-unknown text has no tokens left, so the row is zero.
    assert_eq!(loaded.encode_one("zzz").unwrap(), vec![0.0, 0.0]);
}

#[test]
fn weights_multiply_each_token_row() {
    let rows: [[f32; 2]; 5] = [[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 0.0], [0.0, 0.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let weights = [1.0_f32, 1.0, 0.5, 1.0, 1.0];
    let model = write_model(
        r#"{"normalize": false}"#,
        &[
            ("embeddings", "F32", vec![5, 2], f32_bytes(&flat)),
            ("weights", "F32", vec![5], f32_bytes(&weights)),
        ],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    assert!(loaded.info().has_weights);
    // [1, 2] with weights [1, 0.5] -> ([2,0] + [0,2]) / 2 = [1, 1].
    assert_close(&loaded.encode_one("a b").unwrap(), &[1.0, 1.0]);
}

#[test]
fn mapping_remaps_token_ids_onto_clusters() {
    let rows: [[f32; 2]; 5] = [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 0.0], [0.0, 0.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    // Token 1 and 2 are clustered onto rows 2 and 1 respectively.
    let mapping: [i32; 5] = [0, 2, 1, 0, 0];
    let mapping_bytes: Vec<u8> = mapping.iter().flat_map(|v| v.to_le_bytes()).collect();
    let model = write_model(
        r#"{"normalize": false}"#,
        &[
            ("embeddings", "F32", vec![5, 2], f32_bytes(&flat)),
            ("mapping", "I32", vec![5], mapping_bytes),
        ],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    assert!(loaded.info().has_mapping);
    // "a b" -> ids [1, 2] -> rows [2, 1] -> mean [0, 3].
    assert_close(&loaded.encode_one("a b").unwrap(), &[0.0, 3.0]);
}

#[test]
fn max_length_comes_from_config() {
    let rows: [[f32; 2]; 5] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let tensors = [("embeddings", "F32", vec![5, 2], f32_bytes(&flat))];

    let truncated = write_model(r#"{"normalize": false, "max_length": 2}"#, &tensors, TOKENIZER);
    let model = StaticModel::load(&truncated.path).unwrap();
    assert_eq!(model.info().max_length, Some(2));
    // "a b c d" -> [1,2,3,4] truncated to [1,2] -> mean [1.5, 0].
    assert_close(&model.encode_one("a b c d").unwrap(), &[1.5, 0.0]);

    let unlimited = write_model(
        r#"{"normalize": false, "max_length": null}"#,
        &tensors,
        TOKENIZER,
    );
    let model = StaticModel::load(&unlimited.path).unwrap();
    assert_eq!(model.info().max_length, None);
    assert_close(&model.encode_one("a b c d").unwrap(), &[2.5, 0.0]);

    let defaulted = write_model(r#"{"normalize": false}"#, &tensors, TOKENIZER);
    let model = StaticModel::load(&defaulted.path).unwrap();
    assert_eq!(model.info().max_length, Some(DEFAULT_MAX_LENGTH));
}

#[test]
fn missing_embeddings_is_an_error() {
    let model = write_model(r#"{}"#, &[], TOKENIZER);
    let error = StaticModel::load(&model.path).err().unwrap();
    assert!(error.contains("missing tensor embeddings"), "{error}");
}

#[test]
fn mapping_supports_i64_and_rejects_out_of_range_rows() {
    let rows: [[f32; 2]; 5] = [[0.0; 2]; 5];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let mapping: [i64; 5] = [0, 1, 2, 3, 4];
    let bytes: Vec<u8> = mapping.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ok = write_model(
        r#"{"normalize": false}"#,
        &[
            ("embeddings", "F32", vec![5, 2], f32_bytes(&flat)),
            ("mapping", "I64", vec![5], bytes),
        ],
        TOKENIZER,
    );
    StaticModel::load(&ok.path).unwrap();

    let mapping: [i32; 5] = [0, 1, 2, 3, 9];
    let bytes: Vec<u8> = mapping.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bad = write_model(
        r#"{"normalize": false}"#,
        &[
            ("embeddings", "F32", vec![5, 2], f32_bytes(&flat)),
            ("mapping", "I32", vec![5], bytes),
        ],
        TOKENIZER,
    );
    let error = StaticModel::load(&bad.path).err().unwrap();
    assert!(error.contains("outside the 5-row table"), "{error}");
}

#[test]
fn int8_tables_are_widened() {
    let flat: Vec<i8> = vec![0, 0, 127, 0, 0, 127];
    let bytes: Vec<u8> = flat.iter().map(|v| *v as u8).collect();
    let model = write_model(
        r#"{"normalize": true}"#,
        &[("embeddings", "I8", vec![3, 2], bytes)],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    // "a b" -> rows [1, 2] = [127, 0] and [0, 127] -> mean [63.5, 63.5].
    let expected = std::f32::consts::FRAC_1_SQRT_2;
    assert_close(
        &loaded.encode_one("a b").unwrap(),
        &[expected, expected],
    );
}

#[test]
fn encode_into_checks_the_output_buffer() {
    let rows: [[f32; 2]; 5] = [[0.0; 2]; 5];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let model = write_model(
        r#"{"normalize": false}"#,
        &[("embeddings", "F32", vec![5, 2], f32_bytes(&flat))],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    let mut short = vec![0.0_f32; 2];
    assert!(loaded.encode_into(&["a", "b"], &mut short).is_err());

    let mut out = vec![0.0_f32; 4];
    loaded.encode_into(&["a", "b"], &mut out).unwrap();
    assert_eq!(out, vec![0.0; 4]);
}

/// The real model is not part of the repo; when it is present, check the
/// loader against the file the Python package actually ships.
#[test]
fn loads_the_local_potion_model_when_present() {
    let Ok(dir) = std::env::var("RGREP_M2V_MODEL") else {
        return;
    };
    let model = StaticModel::load(Path::new(&dir)).unwrap();
    let info = model.info();
    assert_eq!(info.dim, 256, "{info:?}");
    assert_eq!(info.dtype, "float16");
    let vector = model.encode_one("parse config file").unwrap();
    let norm = vector.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
}

#[test]
fn f16_round_trip_through_the_converter() {
    for value in [0.0_f32, 1.0, -2.5, 0.333, 65504.0, 1e-5, -0.0] {
        let bits = f32_to_f16(value);
        let back = f16_to_f32(bits);
        if value.abs() > 1e-4 {
            assert!((back - value).abs() <= value.abs() * 1e-3 + 1e-6, "{value} -> {back}");
        } else {
            assert!(back.abs() < 1e-3);
        }
    }
}

#[test]
fn info_reports_the_table_shape() {
    let rows: [[f32; 2]; 5] = [[0.0; 2]; 5];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let model = write_model(
        r#"{"normalize": true}"#,
        &[("embeddings", "F32", vec![5, 2], f32_bytes(&flat))],
        TOKENIZER,
    );
    let loaded = StaticModel::load(&model.path).unwrap();
    let info = loaded.info();
    assert_eq!((info.dim, info.rows), (2, 5));
    assert!(info.normalize);
}
