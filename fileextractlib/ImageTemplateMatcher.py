import cv2
import numpy as np
import PIL.Image as Image


class ImageTemplateMatcher:
    def __init__(self,
                 template: Image.Image,
                 scaling_factor: float = 1.0,
                 enable_multi_scale_matching: bool = False,
                 multi_scale_matching_max_scale: float = 3.5,
                 multi_scale_matching_steps: int = 20):
        if scaling_factor != 1.0:
            template = template.resize((int(template.width * scaling_factor), int(template.height * scaling_factor)))

        self.cv_template = cv2.cvtColor(np.array(template.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.scaling_factor = scaling_factor
        self.template_size = (self.cv_template.shape[:2][1], self.cv_template.shape[:2][0])
        self.enable_multi_scale_matching = enable_multi_scale_matching
        self.multi_scale_matching_max_scale = multi_scale_matching_max_scale
        self.multi_scale_matching_steps = multi_scale_matching_steps

    def match(self, matching_image: Image.Image) -> float:
        # The correlation coefficient matching method always returns a similarity score of 1 if the template image is
        # completely empty (aka black). Obviously that's not correct, so check if the template is empty and return
        # 0 in that case.
        if np.sum(cv2.sumElems(self.cv_template)) == 0:
            return 0.0

        if self.scaling_factor != 1.0:
            matching_image = matching_image.resize(
                (int(matching_image.width * self.scaling_factor), int(matching_image.height * self.scaling_factor)))

        cv_image = cv2.cvtColor(np.array(matching_image.convert("RGB")), cv2.COLOR_RGB2BGR)

        min_scale = max(self.cv_template.shape[0] / cv_image.shape[0],
                        self.cv_template.shape[1] / cv_image.shape[1])

        if self.enable_multi_scale_matching:
            scales_to_match = np.linspace(min_scale,
                                          self.multi_scale_matching_max_scale,
                                          self.multi_scale_matching_steps)
        else:
            scales_to_match = [1.0]

        total_max = 0
        for scale in reversed(scales_to_match):
            resized_template = cv2.resize(self.cv_template,
                                          (int(self.cv_template.shape[1] / scale),
                                           int(self.cv_template.shape[0] / scale)),
                                          interpolation=cv2.INTER_AREA)

            result = cv2.matchTemplate(cv_image, resized_template, cv2.TM_CCOEFF_NORMED)
            (min_val, max_val, _, _) = cv2.minMaxLoc(result)

            if max_val > total_max:
                total_max = max_val

        return total_max


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("template_file")
    parser.add_argument("image_file")
    parser.add_argument("--scaling", type=float, default=1.0)

    args = parser.parse_args()

    start_time = time.time()

    template_file = args.template_file
    image_file = args.image_file

    template = Image.open(template_file)
    image = Image.open(image_file)

    template = template.crop(
        (template.width / 6, template.height / 10, template.width * 5 / 6, template.height * 9 / 10))

    matcher = ImageTemplateMatcher(template=template,
                                   scaling_factor=args.scaling,
                                   enable_multi_scale_matching=True,
                                   multi_scale_matching_steps=20,
                                   multi_scale_matching_max_scale=3.5)

    print("Similarity: ", matcher.match(image))
    print("Matching took: " + str(time.time() - start_time) + " seconds.")
