---
noteId: "152bb7b7275711f0a2f00d8dbde386d7"
tags: []

---

# OCR

Tasks that require to extract the data in a structured format given an image

## Example

A sample query in a zero-shot setting:

```
Input:
During the design challenge, there was a board displaying different materials and their corresponding quantities and weights. A photograph was taken for record-keeping. However, the image is slightly blurred and the handwriting varies in style and size. Examine the image and provide a clear and accurate transcription of the information displayed, detailing the quantities and weights of the card stock, construction paper, and fabric.

Output:
The image shows a whiteboard with information written in marker. There are three columns, each dedicated to a different material: card stock, construction paper, and fabric. For each material, the quantity (in pieces) and the weight (in grams) are given.

Card Stock

Quantity: 3 pieces
Weight: 13 grams
Construction Paper

Quantity: 2 pieces
Weight: 6 grams
Fabric

Quantity: 32
Weight: 77 grams

The weights appear to have been underlined for emphasis. The handwriting for 'Card Stock' and 'Construction Paper' is in black ink, with the numbers also written in black. The word 'Fabric' and its corresponding numbers are written in blue ink. The font styles are casual and varied in size, but the information is nonetheless legible. The whiteboard also has a printed form at the top right corner, but the text in this form is not legible due to the angle and distance at which the photo was taken.
```

## What is the task trying to measure?

Tasks that require to extract the data in a structured format given an image

## Motivation

To verify whether VLM properly understands text within an image.

## Related work

https://arxiv.org/abs/2311.16502
https://arxiv.org/abs/1904.08920