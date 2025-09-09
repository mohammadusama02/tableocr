# -------------------------
# Imports
# -------------------------
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from paddleocr import PaddleOCR, LayoutDetection


# -------------------------
# Step 1: OCR with DocTR
# -------------------------
doc = DocumentFile.from_images("./data/page1.png")
ocr_model = ocr_predictor(pretrained=True)
ocr_result = ocr_model(doc)

ocr_text_boxes = []
for page in ocr_result.pages:
    h, w = page.dimensions
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                (x_min, y_min), (x_max, y_max) = word.geometry
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                ocr_text_boxes.append({
                    "text": word.value,
                    "bbox": (x_min, y_min, x_max, y_max)
                })


# Step 2: Layout detection (tables etc.)
# -------------------------
layout_model = LayoutDetection(model_name="PP-DocLayout_plus-L")
layout_output = layout_model.predict("./data/page1.png", batch_size=1, layout_nms=True)

table_boxes = []
for res in layout_output:
    for box in res["boxes"]:
        if box["label"] == "table":
            coords = list(map(int, box["coordinate"]))  # [x1,y1,x2,y2]
            table_boxes.append(tuple(coords))                


# -------------------------
# Step 3: Filter DocTR OCR text outside table boxes
# -------------------------
def inside_table(box, tables):
    x1, y1, x2, y2 = box
    for tx1, ty1, tx2, ty2 in tables:
        if x1 >= tx1 and y1 >= ty1 and x2 <= tx2 and y2 <= ty2:
            return True
    return False

filtered_texts = [t for t in ocr_text_boxes if not inside_table(t["bbox"], table_boxes)]



# -------------------------
# Step 4: Extract table contents with DocTR
# -------------------------

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize the predictor (use 'db_resnet50' for detection, 'crnn_vgg16_bn' for recognition; or 'vit' for better accuracy)
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
# For GPU: model = ocr_predictor(..., device='cuda:0')
print("DocTR model loaded.")



import cv2
img = cv2.imread("./data/page1.png")
table_contents = []
for idx, tb in enumerate(table_boxes):
    x1, y1, x2, y2 = tb
    crop = img[y1:y2, x1:x2]

    # Run OCR
    result = model([crop])  # Input: np.array (H, W, 3) or DocumentFile.from_images('path')
    json_result = result.export()  # Dict: {'pages': [{'blocks': [{'lines': [{'words': [{'value': text, 'confidence': score, 'geometry': [x0,y0,x1,y1]}]}]}]}]}

    # Extract words with boxes and scores
    words = []
    for page in json_result['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    text = word['value'].strip()
                    if text:  # Skip empty
                        score = word['confidence']
                        geometry = np.array(word['geometry'])  # [[x0,y0], [x1,y1]] for word bbox (quad? but often bilinear)
                        # Convert to polygon if needed (DocTR uses bilinear quads, but for simplicity, use corners)
                        poly = np.array(geometry).reshape(2, 2) * np.array([crop.shape[1], crop.shape[0]])  # Scale to image coords if normalized
                        words.append({'text': text, 'score': score, 'poly': poly})

    print(f"Detected {len(words)} words in table {idx+1}.")

    if len(words) == 0:
        markdown_table = "No text detected."
    else:
        # Filter low-confidence (optional)
        words = [w for w in words if w['score'] > 0.5]

        # Group by row (y-midpoint)
        rows = defaultdict(list)
        for word in words:
            poly = word['poly']  # (4,2) or (2,2); use corners
            if len(poly.shape) == 1:  # If flattened, reshape
                poly = poly.reshape(2, 2)
            y_coords = poly[:, 1]  # Y coordinates
            x_coords = poly[:, 0]  # X coordinates
            y_mid = int(np.mean(y_coords))
            x_mid = int(np.mean(x_coords))
            row_key = y_mid // 10  # Tolerance: 10 pixels; adjust for row spacing

            rows[row_key].append((x_mid, word['text'], word['score']))

        # Sort rows top-to-bottom and words left-to-right
        sorted_rows = []
        for k in sorted(rows.keys()):
            row_list = sorted(rows[k], key=lambda item: item[0])  # Sort by x_mid
            sorted_rows.append([item[1] for item in row_list])  # Just texts

        # Generate Markdown table (pad for alignment)
        if sorted_rows:
            max_cols = max(len(row) for row in sorted_rows)
            padded_rows = [row + [""] * (max_cols - len(row)) for row in sorted_rows]
            header = "| " + " | ".join(["Col"] * max_cols) + " |"
            separator = "| " + " | ".join(["---"] * max_cols) + " |"
            body = "\n".join(["| " + " | ".join(row) + " |" for row in padded_rows])
            markdown_table = f"{header}\n{separator}\n{body}"
        else:
            markdown_table = "No rows formed."

    # Store result
    table_contents.append({
        "bbox": tb,
        "markdown": markdown_table
    })
    print(f"Table {idx+1} Markdown:\n{markdown_table}\n")

    # Save Markdown
    with open(f'table_{idx+1}_output.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)

    # Visualize crop with boxes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Crop")
    ax1.axis('off')

    ax2.imshow(crop)
    for word in words:
        poly = word['poly'].astype(int)
        # Draw rectangle (approx; for quad use Polygon)
        rect = plt.Rectangle((poly[0][0], poly[0][1]), poly[1][0] - poly[0][0], poly[1][1] - poly[0][1], 
                            linewidth=1, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        # Label
        x_text, y_text = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        ax2.text(x_text, y_text, word['text'], fontsize=8, ha='center', color='blue')
    ax2.set_title("OCR with Boxes")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'table_{idx+1}_vis.png', dpi=150)
    plt.show()



# -------------------------
# Step 5: Merge text + tables in reading order
# -------------------------
merged = [{"type": "text", "text": t["text"], "bbox": t["bbox"]} for t in filtered_texts]
merged += [{"type": "table", "text": tbl["markdown"], "bbox": tbl["bbox"]} for tbl in table_contents]

# Sort top-to-bottom, left-to-right
merged.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))



# -------------------------
# Step 6: Print final output
# -------------------------
final_output = []
for item in merged:
    if item["type"] == "text":
        final_output.append(item["text"])
    else:
        final_output.append("\n" + item["text"] + "\n")

print(" ".join(final_output))

