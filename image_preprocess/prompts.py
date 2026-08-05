"""可制造性底座图像编辑提示词。"""


PEDESTAL_PROMPT = """Edit only the transparent masked lower region of the input image.
Keep the character's face, hair, clothing, colors, pose, proportions, accessories,
camera view, and illustration style unchanged. Add one simple, centered, solid
circular display pedestal beneath the character. The feet or lowest garment edge
must visibly overlap and embed into the pedestal by about 5 percent so the generated
3D shape is likely to be connected. The pedestal should be about 80 percent of the
character width and about 6 percent of the character height, with a clearly visible
top surface and side thickness. Preserve a transparent background. Do not add text,
logos, shadows, extra characters, floating decorations, or complex ornamentation."""


def pedestal_prompt() -> str:
    return PEDESTAL_PROMPT

