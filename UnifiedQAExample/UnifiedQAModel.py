from transformers import AutoTokenizer, T5ForConditionalGeneration

def unifiedqa_model_loader(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def unified_qa_prediction(model, tokenizer, question: str, context: str, **generator_args):
    input_text = question + "\\n " + context
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    res = model.generate(input_ids=input_ids, **generator_args)
    answer = tokenizer.batch_decode(res, skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    model_name = "allenai/unifiedqa-t5-large"
    unifiedqa_model, tokeinizer = unifiedqa_model_loader(model_name=model_name)
    for name, param in unifiedqa_model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('-' * 100)