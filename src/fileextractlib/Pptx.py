def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    text_runs = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)

    return text_runs

file_path = 'path_to_your_pptx_file.pptx'
text_content = extract_text_from_pptx(file_path)
for text in text_content:
    print(text)
