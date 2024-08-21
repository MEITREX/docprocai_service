import cv2
import numpy as np
import PIL.Image as Image


class ImageTemplateMatcher:
    def __init__(self,
                 template: Image.Image,
                 threshold: float = 0.8,
                 enable_multi_scale_matching: bool = False,
                 multi_scale_matching_max_scale: float = 3.0,
                 multi_scale_matching_steps: int = 20):
        self.cv_template = cv2.cvtColor(np.array(template.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.template_size = (self.cv_template.shape[:2][1], self.cv_template.shape[:2][0])
        self.threshold = threshold
        self.enable_multi_scale_matching = enable_multi_scale_matching
        self.multi_scale_matching_max_scale = multi_scale_matching_max_scale
        self.multi_scale_matching_steps = multi_scale_matching_steps

    def match(self, image: Image.Image) -> float:
        cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        min_scale = max(self.cv_template.shape[0] / cv_image.shape[0],
                        self.cv_template.shape[1] / cv_image.shape[1])

        if self.enable_multi_scale_matching:
            scales_to_match = np.linspace(min_scale,
                                          self.multi_scale_matching_max_scale,
                                          self.multi_scale_matching_steps)
        else:
            scales_to_match = [1.0]

        total_max = 0
        for scale in scales_to_match:
            resized = cv2.resize(cv_image,
                                 (int(cv_image.shape[1] * scale), int(cv_image.shape[0] * scale)),
                                 interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(resized, self.cv_template, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

            if max_val > total_max:
                total_max = max_val

            # if the resized image is smaller than the template, then break from the loop
            if resized.shape[0] < self.template_size[0] or resized.shape[1] < self.template_size[1]:
                break

        return total_max


if __name__ == "__main__":
    import sys

    template_file = sys.argv[1]
    image_file = sys.argv[2]

    template = Image.open(template_file)

    template = template.crop((template.width / 6, template.height / 10, template.width * 5 / 6, template.height * 9 / 10))

    matcher = ImageTemplateMatcher(template=template)
    print(matcher.match(Image.open(image_file)))
