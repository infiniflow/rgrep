/* rgrep_native — tree-sitter code splitting, HF tokenization and static
 * (Model2Vec) embeddings.
 *
 * The JSON calls return a string that the caller must release with rg_free.
 * On failure they return NULL and, when the last argument is not NULL, store a
 * message there (also released with rg_free).
 *
 * The JSON schema version is reported by rg_abi_version and must match
 * internal/native.ABIExpected on the Go side; the model2vec handle API was
 * added without a schema change, so it reports the same version.
 */
#ifndef RGREP_NATIVE_H
#define RGREP_NATIVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* {"language": "...", "degraded": false, "truncated": false,
 *  "units": [{"kind": "function", "name": "f", "parent": "",
 *             "signature": "func f(a int) error", "doc": "...",
 *             "calls": ["g"], "start_line": 3, "end_line": 5,
 *             "start_byte": 12, "end_byte": 48}]} */
char *rg_codesplit(const char *lang, const char *path, const char *content,
                   size_t content_len, char **err_out);

/* {"ids": [1, 2, 3]} — no special tokens are added. */
char *rg_tokenize(const char *tokenizer_path, const char *text, size_t text_len,
                  char **err_out);

/* JSON array of the languages rg_codesplit accepts. */
char *rg_languages(void);

/* Frees a string returned by this library. NULL is accepted. */
void rg_free(char *ptr);

/* JSON schema version of rg_codesplit. */
int rg_abi_version(void);

/* --- static embeddings (Model2Vec) --------------------------------------
 *
 * A handle owns a loaded table + tokenizer; loading is once per process.
 * `rg_m2v_encode` writes count * rg_m2v_dim(handle) floats into `out`,
 * row-major, and returns 0 on success. `lens` may be NULL, in which case the
 * texts must be NUL-terminated.
 */
typedef struct RgModelHandle RgModelHandle;

RgModelHandle *rg_m2v_load(const char *model_dir, char **err_out);
size_t rg_m2v_dim(const RgModelHandle *handle);
/* {"dim":256,"rows":63457,"vocab":63457,"dtype":"float16","normalize":true,
 *  "max_length":512,"has_weights":false,"has_mapping":false} */
char *rg_m2v_info(const RgModelHandle *handle);
int rg_m2v_encode(const RgModelHandle *handle, const char *const *texts,
                  const size_t *lens, size_t count, float *out,
                  char **err_out);
void rg_m2v_free(RgModelHandle *handle);

#ifdef __cplusplus
}
#endif

#endif /* RGREP_NATIVE_H */
