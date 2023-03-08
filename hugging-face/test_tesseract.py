from transformers import pipeline

vqa = pipeline(model="impira/layoutlm-document-qa")
print(vqa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?",
)
)
