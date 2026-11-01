use anyhow::Result;
use oxide_infer_server::OxideTokenizer;

#[test]
fn test_lfm3_tokenizer_loads_and_encodes() -> Result<()> {
    let tokenizer_path = "/mnt/hdd/models/LFM3-VL-2B/tokenizer.json";
    let tokenizer = OxideTokenizer::from_file(tokenizer_path)?;

    let prompt = "Explain the structure of this image:";
    let token_ids = tokenizer.encode(prompt)?;
    assert!(!token_ids.is_empty(), "Token IDs should not be empty");

    let decoded = tokenizer.decode(&token_ids)?;
    assert_eq!(decoded.trim(), prompt.trim());

    println!("LFM3 Token IDs for '{}': {:?}", prompt, token_ids);
    Ok(())
}
