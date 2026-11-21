/* rgrep_native — tree-sitter code splitting and HF tokenization.
 *
 * Both calls return a JSON string that the caller must release with rg_free.
 * On failure they return NULL and, when the last argument is not NULL, store a
 * message there (also released with rg_free).
 *
 * The JSON schema version is reported by rg_abi_version and must match
 * internal/native.ABIExpected on the Go side.
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

#ifdef __cplusplus
}
#endif

#endif /* RGREP_NATIVE_H */
