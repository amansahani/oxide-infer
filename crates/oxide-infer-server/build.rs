fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");
}
