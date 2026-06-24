//! Exercise the `extern "C"` surface directly (as Rust calls) — a CI-portable proof the ABI is wired
//! end-to-end, including the function-pointer host-capability callback. A real C program linking the
//! staticlib is in `examples/` (built with `cc`, see `examples/README.md`).

use super::*;
use std::ffi::CString;

// Two C-ABI host capabilities: `add_seven(x) = x + 7` and `triple(x) = x * 3`.
extern "C" fn add_seven(
    _ctx: *mut c_void,
    _op: u32,
    args: *const i64,
    n_args: usize,
    results: *mut i64,
    cap: usize,
) -> i32 {
    if n_args < 1 || cap < 1 {
        return -1;
    }
    unsafe {
        *results = *args + 7;
    }
    1
}
extern "C" fn triple(
    _ctx: *mut c_void,
    _op: u32,
    args: *const i64,
    n_args: usize,
    results: *mut i64,
    cap: usize,
) -> i32 {
    if n_args < 1 || cap < 1 {
        return -1;
    }
    unsafe {
        *results = *args * 3;
    }
    1
}

const NAMED: &str = "\
memory 15
export \"entry\" 0
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 5
  v4 = call.import \"add_seven\" (i64) -> (i64) v2 (v3)
  v5 = i64.const 4
  v6 = i32.load v5
  v7 = call.import \"triple\" (i64) -> (i64) v6 (v4)
  return v7
}
";

#[test]
fn name_bound_host_fn_callbacks_run_through_the_c_abi() {
    unsafe {
        let ir = CString::new(NAMED).unwrap();
        let m = svm_module_parse_text(ir.as_ptr());
        assert!(!m.is_null(), "parse");
        assert_eq!(svm_module_synth_powerbox_start(m, 0, 2, false), SVM_OK);

        let imports = svm_imports_new();
        let n_add = CString::new("add_seven").unwrap();
        let n_tri = CString::new("triple").unwrap();
        assert_eq!(
            svm_imports_provide_host_fn(imports, n_add.as_ptr(), 0, add_seven, ptr::null_mut()),
            SVM_OK
        );
        assert_eq!(
            svm_imports_provide_host_fn(imports, n_tri.as_ptr(), 0, triple, ptr::null_mut()),
            SVM_OK
        );

        // Consumes `m` and `imports`.
        let inst = svm_instantiate_with_imports(m, imports);
        assert!(!inst.is_null(), "instantiate by name");

        let run = svm_instance_run_diff(inst, ptr::null());
        assert!(!inst.is_null());
        assert!(!run.is_null(), "run_diff");

        assert_eq!(svm_run_outcome_kind(run), SVM_OUTCOME_RETURNED);
        assert_eq!(svm_run_result_count(run), 1);
        assert_eq!(
            svm_run_result(run, 0),
            36,
            "(5 + 7) * 3 across the C callbacks"
        );

        svm_run_free(run);
        svm_instance_free(inst);
    }
}

const HELLO: &str = "\
memory 15
data ro 16384 \"hi from C\\n\"
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 16384
  v4 = i64.const 10
  v5 = call.import \"write\" (i64, i64) -> (i64) v2 (v3, v4)
  v6 = i32.const 0
  return v6
}
";

#[test]
fn builtin_stdout_and_each_backend_via_c_abi() {
    unsafe {
        for backend in [SVM_BACKEND_TREEWALK, SVM_BACKEND_BYTECODE, SVM_BACKEND_JIT] {
            let ir = CString::new(HELLO).unwrap();
            let m = svm_module_parse_text(ir.as_ptr());
            assert_eq!(svm_module_synth_powerbox_start(m, 0, 1, false), SVM_OK);
            let imports = svm_imports_new();
            let n_write = CString::new("write").unwrap();
            assert_eq!(
                svm_imports_provide_stdout(imports, n_write.as_ptr()),
                SVM_OK
            );
            let inst = svm_instantiate_with_imports(m, imports);
            assert!(!inst.is_null(), "instantiate (backend {backend})");

            let run = svm_instance_run(inst, backend, ptr::null());
            assert!(!run.is_null(), "run backend {backend}");

            let mut len = 0usize;
            let p = svm_run_stdout(run, &mut len);
            let out = std::slice::from_raw_parts(p, len);
            assert_eq!(out, b"hi from C\n", "stdout on backend {backend}");

            svm_run_free(run);
            svm_instance_free(inst);
        }
    }
}

#[test]
fn run_config_threads_fuel_and_memory() {
    unsafe {
        // fuel=1 out-of-fuels the tree-walker; the JIT ignores it.
        let cfg = SvmRunConfig {
            fuel: 1,
            fuel_set: 1,
            deadline_ms: 0,
            deadline_set: 0,
            max_fibers: 0,
            max_vcpus: 0,
            stdin: ptr::null(),
            stdin_len: 0,
            memory_size_log2: 0,
            memory_set: 0,
        };
        let mk = || {
            let ir = CString::new(HELLO).unwrap();
            let m = svm_module_parse_text(ir.as_ptr());
            assert_eq!(svm_module_synth_powerbox_start(m, 0, 1, false), SVM_OK);
            let imports = svm_imports_new();
            let n = CString::new("write").unwrap();
            assert_eq!(svm_imports_provide_stdout(imports, n.as_ptr()), SVM_OK);
            svm_instantiate_with_imports(m, imports)
        };

        let inst = mk();
        let trapped = svm_instance_run(inst, SVM_BACKEND_TREEWALK, &cfg);
        assert!(trapped.is_null(), "fuel=1 must out-of-fuel the tree-walker");
        assert!(!svm_last_error().is_null(), "an error message was set");
        svm_instance_free(inst);

        let inst = mk();
        let ok = svm_instance_run(inst, SVM_BACKEND_JIT, &cfg);
        assert!(!ok.is_null(), "the JIT ignores per-op fuel");
        svm_run_free(ok);
        svm_instance_free(inst);
    }
}

#[test]
fn errors_are_fail_closed_not_panics() {
    unsafe {
        // Bad IR → null + an error message, no panic.
        let bad = CString::new("this is not IR {{{").unwrap();
        assert!(svm_module_parse_text(bad.as_ptr()).is_null());
        assert!(!svm_last_error().is_null());

        // Null handles are tolerated.
        assert_eq!(
            svm_module_synth_powerbox_start(ptr::null_mut(), 0, 3, false),
            SVM_ERR_FAILED
        );
        assert!(svm_instantiate(ptr::null_mut()).is_null());
        svm_module_free(ptr::null_mut()); // no-op, no crash
        svm_run_free(ptr::null_mut());

        // An unbound import fails closed at instantiate.
        let ir = CString::new(NAMED).unwrap();
        let m = svm_module_parse_text(ir.as_ptr());
        assert_eq!(svm_module_synth_powerbox_start(m, 0, 2, false), SVM_OK);
        let imports = svm_imports_new(); // empty — neither name bound
        let inst = svm_instantiate_with_imports(m, imports);
        assert!(inst.is_null(), "unbound imports must fail closed");
        assert!(!svm_last_error().is_null());
    }
}
