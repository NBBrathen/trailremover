# backend/tests/unit_tests.py

import unittest
from unittest.mock import patch
import numpy as np
import sys
import pathlib
import importlib
from astropy.io import fits

# -------------------------------------------------------------------
# Make project imports work no matter where tests are run from
# -------------------------------------------------------------------

TEST_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = TEST_FILE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Some parts of the code import "app.*", but the real location is backend.app.
# This maps the name so the imports don't break during testing.
try:
    import app
except ModuleNotFoundError:
    backend_app = importlib.import_module("backend.app")
    sys.modules["app"] = backend_app

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from backend.app.models.job import JobStatus, JobResponse, CreateJobResponse
from backend.app.models.trail import Trail, DetectionResponse, CorrectionRequest
from backend.app.services.job_manager import JobManager, JobStatus as JMStatus
from backend.app.services.image_processor import ImageProcessor
from backend.app.core.restoration import (
    restore_pixels,
    restore_pixels_iterative,
    evaluate_restoration_quality,
)
from backend.app.core import detection as detmod
from backend.app.api.v1.routes import api_router


# -------------------------------------------------------------------
# Tests for models
# -------------------------------------------------------------------

class TestModels(unittest.TestCase):
    def test_job_models(self):
        # Check that creating the job model works and fields are correct
        j = JobResponse(
            job_id="abc",
            status=JobStatus.QUEUED,
            filename="file.fits",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )

        self.assertEqual(j.status, JobStatus.QUEUED)
        self.assertIn("file.fits", j.model_dump_json())

        c = CreateJobResponse(job_id="xyz")
        self.assertEqual(c.job_id, "xyz")
        self.assertEqual(c.message, "Job created successfully")

    def test_trail_models(self):
        # Check that trail-related models store values correctly
        t = Trail(
            trail_id="t1",
            start_point=(1, 2),
            end_point=(3, 4),
            width=2.5,
            confidence=0.7,
        )
        self.assertEqual(t.end_point, (3, 4))

        dr = DetectionResponse(job_id="j1", trail_count=1, trails=[t])
        self.assertEqual(dr.trail_count, 1)

        cr = CorrectionRequest(trails_to_correct=["t1", "t2"])
        self.assertEqual(len(cr.trails_to_correct), 2)


# -------------------------------------------------------------------
# Tests for JobManager
# -------------------------------------------------------------------

class TestJobManager(unittest.TestCase):
    def test_job_lifecycle(self):
        # Create a job → change status → attach detected trails
        jm = JobManager()
        job_id = jm.create_job("x.fits", pathlib.Path("x.fits"))

        job = jm.get_job(job_id)
        self.assertEqual(job["status"], JMStatus.QUEUED)

        jm.update_job_status(job_id, JMStatus.DETECTING, note="running")
        self.assertEqual(jm.get_job(job_id)["status"], JMStatus.DETECTING)

        jm.set_detected_trails(job_id, [{"trail_id": "a"}])
        self.assertEqual(jm.get_job(job_id)["detected_trails"][0]["trail_id"], "a")


# -------------------------------------------------------------------
# Tests for ImageProcessor (FITS handling + restoration wrapper)
# -------------------------------------------------------------------

class TestImageProcessor(unittest.TestCase):
    def test_load_and_save_fits(self):
        # Write a small FITS file, load it, and save a new one
        ip = ImageProcessor()

        tmp_in = pathlib.Path("test_input.fits")
        data = np.ones((4, 4), dtype=np.float32)
        fits.PrimaryHDU(data).writeto(tmp_in, overwrite=True)

        loaded = ip.load_fits_image(tmp_in)
        self.assertEqual(loaded.shape, (4, 4))

        tmp_out = pathlib.Path("test_output.fits")
        ip.save_fits_image(loaded, tmp_out, original_path=tmp_in)
        self.assertTrue(tmp_out.exists())

        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)

    def test_apply_restoration_basic(self):
        # Should run normally whether or not trails are provided
        ip = ImageProcessor()
        img = np.zeros((16, 16), dtype=np.float32)

        out1 = ip.apply_restoration(img, [])
        self.assertEqual(out1.shape, img.shape)

        mask = np.zeros_like(img)
        mask[4:8, 4:12] = 1.0

        out2 = ip.apply_restoration(img, [{"mask": mask}])
        self.assertEqual(out2.shape, img.shape)


# -------------------------------------------------------------------
# Tests for restoration logic
# -------------------------------------------------------------------

class TestRestoration(unittest.TestCase):
    def test_restore_pixels_shape(self):
        # Output size should match input size
        img = np.zeros((10, 10), np.float32)
        m1 = np.zeros_like(img); m1[2:4, 2:8] = 1
        m2 = np.zeros_like(img); m2[6:8, 1:9] = 1
        trails = [{"mask": m1}, {"mask": m2}]

        out = restore_pixels(img, trails, method="telea", inpaint_radius=5)
        self.assertEqual(out.shape, img.shape)

    def test_iterative_restoration_calls_base(self):
        # Test that iterative restoration calls restore_pixels multiple times
        img = np.zeros((10, 10))
        m = np.zeros_like(img); m[4, :] = 1

        with patch("backend.app.core.restoration.restore_pixels",
                   side_effect=lambda *a, **k: a[0]) as base:
            restore_pixels_iterative(img, [{"mask": m}], iterations=3)
            self.assertEqual(base.call_count, 3)

    def test_quality_metrics(self):
        # Make sure quality metrics return expected fields
        img = np.zeros((10, 10), np.float32)
        restored = img.copy()
        mask = np.zeros_like(img); mask[3:7, 3:7] = 255

        metrics = evaluate_restoration_quality(img, restored, mask)
        expected = {
            "mean_absolute_difference",
            "restored_mean",
            "restored_std",
            "surrounding_mean",
            "surrounding_std",
            "noise_ratio",
        }
        self.assertTrue(expected.issubset(metrics))


# -------------------------------------------------------------------
# Tests for detection functions
# -------------------------------------------------------------------

class TestDetection(unittest.TestCase):
    def test_preprocess_and_mask(self):
        # Make sure preprocessing normalizes correctly and masks are created
        arr = np.linspace(0, 1000, 100).reshape(10, 10).astype(np.float32)
        rgb = detmod.preprocess_fits_data(arr)

        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertGreaterEqual(rgb.min(), 0)
        self.assertLessEqual(rgb.max(), 1)

        mask = detmod.create_trail_mask(
            arr.shape,
            contour=[[1, 1], [8, 1], [8, 8], [1, 8]],
            confidence=1.0,
        )
        self.assertEqual(mask.shape, arr.shape)
        self.assertGreater(mask.sum(), 0)

    def test_detect_trails_single(self):
        # Fake detector to test non-tiled branch
        class FakeDet:
            confidence_threshold = 0.5
            min_trail_pixels = 5

            def detect(self, rgb, return_mask=False):
                h, w = rgb.shape[:2]
                contour = [[1, 1], [w-2, 1], [w-2, 2], [1, 2]]
                return {
                    "num_trails": 1,
                    "trails": [{
                        "start_point": (1, 1),
                        "end_point": (w-2, 2),
                        "width": 2.0,
                        "confidence": 0.9,
                        "contour": contour,
                    }],
                    "mask": np.zeros((h, w), dtype=np.float32),
                }

        with patch.object(detmod, "get_detector", return_value=FakeDet()):
            img = np.zeros((64, 64), np.float32)
            trails = detmod.detect_trails(img)

        t = trails[0]
        for key in ("trail_id", "start_point", "end_point", "width", "confidence", "mask"):
            self.assertIn(key, t)

    def test_detect_trails_tiled(self):
        # Large image forces tiled branch
        with patch.object(detmod, "_detect_trails_tiled", return_value=[]) as tiled, \
             patch.object(detmod, "get_detector", return_value=object()):
            big = np.zeros((2048, 2048), np.float32)
            out = detmod.detect_trails(big)

        self.assertEqual(out, [])
        tiled.assert_called_once()


# -------------------------------------------------------------------
# Router import test
# -------------------------------------------------------------------

class TestRouting(unittest.TestCase):
    def test_router_exists(self):
        # Just checking that the router loads correctly
        self.assertIsNotNone(api_router)
        self.assertGreaterEqual(len(api_router.routes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
