import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class DashboardComparator:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def compare(self, image_path_1, image_path_2):
        image_1 = self._load_image(image_path_1)
        image_2 = self._load_image(image_path_2)

        if image_1 is None or image_2 is None:
            raise ValueError(f"Failed to load images: {image_path_1}, {image_path_2}")

        height = max(image_1.shape[0], image_2.shape[0])
        width = max(image_1.shape[1], image_2.shape[1])

        image_1 = cv2.resize(image_1, (width, height))
        image_2 = cv2.resize(image_2, (width, height))

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        similarity_score, difference_map = ssim(gray_1, gray_2, full=True)

        binary_diff = (difference_map < self.threshold).astype(np.uint8) * 255

        changed_pixels = np.sum(binary_diff > 0)
        total_pixels = binary_diff.size
        change_percentage = (changed_pixels / total_pixels) * 100

        colored_diff = cv2.applyColorMap((difference_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_image = cv2.addWeighted(image_1, 0.6, colored_diff, 0.4, 0)

        if similarity_score > 0.95:
            verdict = "Практически идентичны"
        elif similarity_score > 0.85:
            verdict = "Незначительные изменения"
        elif similarity_score > 0.70:
            verdict = "Заметные изменения"
        else:
            verdict = "Сильно отличаются"

        description = self._generate_description(change_percentage, similarity_score, binary_diff, width, height)

        return {
            "similarity": round(similarity_score, 3),
            "change_percentage": round(change_percentage, 2),
            "difference_map": binary_diff,
            "overlay": overlay_image,
            "verdict": verdict,
            "description": description
        }

    def _load_image(self, image_path):
        with open(image_path, "rb") as file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    def _generate_description(self, change_percentage, similarity_score, binary_diff, width, height):
        description_parts = []

        if change_percentage < 1:
            description_parts.append("Минимальные различия, возможно артефакты сжатия")
        elif change_percentage < 5:
            description_parts.append("Небольшие локальные изменения")
        elif change_percentage < 15:
            description_parts.append("Умеренные изменения")
        else:
            description_parts.append("Существенные изменения макета")

        half_height, half_width = binary_diff.shape
        left_region = np.sum(binary_diff[:, :half_width//3]) > 100
        center_region = np.sum(binary_diff[:, half_width//3:2*half_width//3]) > 100
        right_region = np.sum(binary_diff[:, 2*half_width//3:]) > 100

        changed_regions = []
        if left_region:
            changed_regions.append("левой")
        if center_region:
            changed_regions.append("центральной")
        if right_region:
            changed_regions.append("правой")

        if changed_regions:
            region_text = ", ".join(changed_regions)
            description_parts.append(f"Изменения обнаружены в {region_text} области")

        if similarity_score > 0.9 and change_percentage < 5:
            description_parts.append("Дашборды визуально очень близки")
        elif similarity_score < 0.6:
            description_parts.append("Дашборды выглядят как совершенно разные")

        return ". ".join(description_parts) + "."

    def save_overlay(self, overlay_image, output_path):
        success, encoded_image = cv2.imencode('.png', overlay_image)
        if success:
            with open(output_path, 'wb') as file:
                file.write(encoded_image.tobytes())