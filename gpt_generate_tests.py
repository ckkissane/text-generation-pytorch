import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_greedy_search(_greedy_search):
    vocab_size = 50257
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # use tokenizer.eos_token_id to avoid annoying HF warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    test_cases = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (1, 100)),
    ]
    for input_ids in test_cases:
        hf_greedy_output = model.generate(input_ids, max_new_tokens=10)
        my_greedy_output = _greedy_search(model, input_ids, max_new_tokens=10)

        if torch.allclose(hf_greedy_output, my_greedy_output):
            print("Congrats! you passed the test")
        else:
            print("Your result does not match HuggingFace")

def test_beam_search(_beam_search):
    vocab_size = 50257
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # use tokenizer.eos_token_id to avoid annoying HF warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    # HF generate doesn't handle sequences longer than 1024, so we won't test those
    test_cases = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (1, 20)),
    ]
    for input_ids in test_cases:
        hf_beam_output = model.generate(input_ids, max_new_tokens=5, num_beams=3)
        my_beam_output = _beam_search(model, input_ids, max_new_tokens=5, num_beans=3)
    
        if torch.allclose(hf_beam_output, my_beam_output):
            print("Congrats! you passed the test")
        else:
            print("Your result down not match HuggingFace")

def test_sample(_sample):
    vocab_size = 50257
    seed = 0
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # use tokenizer.eos_token_id to avoid annoying HF warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    # HF generate doesn't handle sequences longer than 1024, so we won't test those
    test_cases = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (1, 20)),
    ]
    for input_ids in test_cases:
        torch.manual_seed(seed)
        hf_sample_output = model.generate(input_ids, do_sample=True, max_new_tokens=5, top_k=0)

        torch.manual_seed(seed)
        my_sample_output = _sample(model, input_ids, max_new_tokens=5)
    
        if torch.allclose(hf_sample_output, my_sample_output):
            print("Congrats! you passed the test")
        else:
            print("Your result down not match HuggingFace")

def test_sample_temperature(_sample_temperature):
    vocab_size = 50257
    seed = 0
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # use tokenizer.eos_token_id to avoid annoying HF warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    # HF generate doesn't handle sequences longer than 1024, so we won't test those
    test_cases = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (1, 20)),
    ]
    for input_ids in test_cases:
        torch.manual_seed(seed)
        hf_sample_output = model.generate(input_ids, do_sample=True, max_new_tokens=5, top_k=0, temperature=0.7)

        torch.manual_seed(seed)
        my_sample_output = _sample_temperature(model, input_ids, max_new_tokens=5, temperature=0.7)
    
        if torch.allclose(hf_sample_output, my_sample_output):
            print("Congrats! you passed the test")
        else:
            print("Your result down not match HuggingFace")

def test_sample_top_k(_sample_top_k):
    vocab_size = 50257
    seed = 0
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # use tokenizer.eos_token_id to avoid annoying HF warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    # HF generate doesn't handle sequences longer than 1024, so we won't test those
    test_cases = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (1, 20)),
    ]
    for input_ids in test_cases:
        torch.manual_seed(seed)
        hf_sample_output = model.generate(input_ids, do_sample=True, max_new_tokens=5, top_k=50)

        torch.manual_seed(seed)
        my_sample_output = _sample_top_k(model, input_ids, max_new_tokens=5, top_k=50)
    
        if torch.allclose(hf_sample_output, my_sample_output):
            print("Congrats! you passed the test")
        else:
            print("Your result down not match HuggingFace")

def test_sample_top_p(_sample_top_p):
    vocab_size = 50257
    seed = 0
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # use tokenizer.eos_token_id to avoid annoying HF warnings
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    # HF generate doesn't handle sequences longer than 1024, so we won't test those
    test_cases = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (1, 20)),
    ]
    for input_ids in test_cases:
        torch.manual_seed(seed)
        hf_sample_output = model.generate(input_ids, do_sample=True, max_new_tokens=5, top_p=0.92, top_k=0)

        torch.manual_seed(seed)
        my_sample_output = _sample_top_p(model, input_ids, max_new_tokens=5, top_p=0.92)
    
        if torch.allclose(hf_sample_output, my_sample_output):
            print("Congrats! you passed the test")
        else:
            print("Your result down not match HuggingFace")
