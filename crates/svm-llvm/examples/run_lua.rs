//! Throwaway: translate a `.ll` and run it through the powerbox on the **tree-walker** (Memory
//! granted), printing the outcome. Validates Lua first light — `main` runs a pure-compute script and
//! returns the result. Pass `jit`/`bytecode` as a 2nd arg to pick another backend.
use svm_run::{Backend, Limits, RunConfig};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run_lua <file.ll> [backend]");
    let backend = match std::env::args().nth(2).as_deref() {
        Some("jit") => Backend::Jit,
        Some("bytecode") => Backend::Bytecode,
        _ => Backend::TreeWalk,
    };
    let t = svm_llvm::translate_ll_path(&path).expect("translate");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    };
    match inst.run(backend, &config) {
        Ok(run) => {
            println!("outcome = {:?}", run.outcome);
            if !run.stdout.is_empty() {
                println!("stdout = {:?}", String::from_utf8_lossy(&run.stdout));
            }
            if !run.stderr.is_empty() {
                println!("stderr = {:?}", String::from_utf8_lossy(&run.stderr));
            }
        }
        Err(e) => println!("ERROR: {e}"),
    }
}
