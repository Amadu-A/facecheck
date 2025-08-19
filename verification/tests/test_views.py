# verification/tests/test_views.py
from unittest import mock
import numpy as np
import cv2
from django.test import TestCase
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from verification.services.yolo_face_detector import DetectedFace

def _fake_png(color=(0, 0, 0), size=(100, 100)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

class ViewsFlowTest(TestCase):
    def test_full_flow_with_mocks(self):
        # 1) upload doc
        doc_bytes = _fake_png()
        r1 = self.client.post(
            reverse("upload_document"),
            data={"document": SimpleUploadedFile("doc.png", doc_bytes, content_type="image/png")},
        )
        self.assertEqual(r1.status_code, 302)

        # 2) upload selfie
        selfie_bytes = _fake_png(color=(255, 255, 255))
        r2 = self.client.post(
            reverse("upload_selfie"),
            data={"selfie": SimpleUploadedFile("selfie.png", selfie_bytes, content_type="image/png")},
        )
        self.assertEqual(r2.status_code, 302)

        # 3) mock models
        with mock.patch("verification.views._ensure_models"), \
             mock.patch("verification.views.MODEL_BUNDLE") as bundle, \
             mock.patch("verification.views.imread", side_effect=lambda p: cv2.imdecode(np.frombuffer(doc_bytes, np.uint8), cv2.IMREAD_COLOR)):

            det_face = DetectedFace((10, 10, 90, 90), 0.95)
            bundle.yolo.detect_best_face.side_effect = [det_face, det_face]  # doc, selfie
            bundle.embedder.embed.side_effect = [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ]

            r3 = self.client.post(reverse("analyze"))
            self.assertEqual(r3.status_code, 302)

            r4 = self.client.get(reverse("home"))
            self.assertContains(r4, "Верификация пройдена")
