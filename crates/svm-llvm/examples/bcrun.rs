use svm_interp::Value;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = svm_llvm::translate_bc_path(std::path::Path::new(&a[1])).expect("tr");
    let sp = t.entry_sp as i64;
    let e = t.exports.iter().find(|(s, _)| s == "run").unwrap().1;
    let n: i64 = a[2].parse().unwrap();
    let mut f = u64::MAX;
    println!(
        "{:?}",
        svm_interp::run(&t.module, e, &[Value::I64(sp), Value::I64(n)], &mut f)
    );
}
