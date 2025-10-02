"""
Face Matching MVP - Gradio Web Application
Duplicate account detection using face matching with Milvus Lite
"""

import gradio as gr
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple, List

from face_matcher.config import config, PROJECT_ROOT
from face_matcher.core.detection import FaceDetectorFactory
from face_matcher.core.recognition import FaceEmbeddingExtractor
from face_matcher.core.database import VectorDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_image_path(stored_path: str) -> str:
    """
    Convert stored absolute path to correct path for current environment.

    Handles paths stored as absolute paths (e.g., /mnt/d/project/face_matching_mvp/data/...)
    and converts them to work in both local and Hugging Face deployments.

    Args:
        stored_path: Path stored in database (may be absolute)

    Returns:
        Resolved absolute path for current environment
    """
    # Convert to Path object
    path = Path(stored_path)

    # Extract the relative portion after 'face_matching_mvp' or use parts after 'data'
    parts = path.parts

    # Find 'data' in path parts and reconstruct from there
    if 'data' in parts:
        data_index = parts.index('data')
        relative_parts = parts[data_index:]
        resolved_path = PROJECT_ROOT / Path(*relative_parts)
        return str(resolved_path)

    # If path is already relative or doesn't contain 'data', return as-is
    return stored_path


class FaceMatchingApp:
    """Main application class for face matching"""

    def __init__(self):
        """Initialize Face Matching Application"""
        logger.info("Initializing Face Matching Application...")

        # Initialize embedding extractor
        logger.info(f"Loading model: {config.model.model_path}")
        self.embedding_extractor = FaceEmbeddingExtractor(
            model_path=config.model.model_path,
            device=config.model.device
        )

        # Initialize detectors (lazy loading)
        self.detectors = {}

        # Initialize vector database
        logger.info(f"Connecting to database: {config.database.db_path}")
        self.database = VectorDatabase(
            db_path=config.database.db_path,
            collection_name=config.database.collection_name,
            embedding_dim=config.database.embedding_dim
        )

        # Load collection if exists
        try:
            self.database.create_collection(drop_existing=False)
            self.database.load_collection()
            stats = self.database.get_stats()
            logger.info(f"Loaded collection with {stats['num_entities']} entities")
            self.database_ready = True
        except Exception as e:
            logger.warning(f"Database not ready: {e}")
            logger.warning("Please run prepare_database.py first")
            self.database_ready = False

    def get_detector(self, detector_type: str):
        """
        Get detector instance (lazy loading)

        Args:
            detector_type: "RetinaFace" or "Haar Cascade"

        Returns:
            Detector instance
        """
        detector_key = detector_type.lower().replace(" ", "")

        if detector_key not in self.detectors:
            logger.info(f"Initializing {detector_type} detector...")
            self.detectors[detector_key] = FaceDetectorFactory.create_detector(detector_key)

        return self.detectors[detector_key]

    def process_image(
        self,
        input_image: np.ndarray,
        detector_type: str
    ) -> Tuple[str, str, List[Tuple[np.ndarray, str]]]:
        """
        Process uploaded image and search for similar faces

        Args:
            input_image: Input image from Gradio (RGB numpy array)
            detector_type: "RetinaFace" or "Haar Cascade"

        Returns:
            Tuple of (status_message, confidence_text, gallery_images)
        """
        if input_image is None:
            return "⚠️ Please upload an image", "", []

        if not self.database_ready:
            return (
                "❌ Database not initialized",
                "Please run `python prepare_database.py` first to populate the database",
                []
            )

        try:
            # Get detector
            detector = self.get_detector(detector_type)

            # Detect and align face
            aligned_face = detector.detect_and_align(input_image)

            if aligned_face is None:
                return (
                    "❌ No face detected in the uploaded image",
                    f"**Detector:** {detector_type}\n\n"
                    "**Tip:** Try a different detector or ensure the face is clearly visible",
                    []
                )

            # Extract embedding
            query_embedding = self.embedding_extractor.extract_embedding(aligned_face)

            # Search for similar faces
            results = self.database.search(
                query_embedding=query_embedding,
                top_k=config.app.top_k_results,
                nprobe=config.database.nprobe
            )

            # Check if no results
            if len(results) == 0:
                return (
                    "✅ No Duplicate Found",
                    "**Status:** No similar faces found in database\n\n"
                    f"**Detector:** {detector_type}",
                    []
                )

            # Get best match
            best_match = results[0]
            best_similarity = best_match['similarity']

            # Count duplicates (faces above threshold)
            duplicate_count = sum(1 for result in results if result['similarity'] >= config.app.similarity_threshold)

            # Determine duplicate status
            if best_similarity >= config.app.similarity_threshold:
                if duplicate_count == 1:
                    status = "🚨 Duplicate Detected (1 match)"
                else:
                    status = f"🚨 Duplicate Detected ({duplicate_count} matches)"
                status_emoji = "🚨"
            else:
                status = "✅ No Duplicate Found"
                status_emoji = "✅"

            # Format confidence text
            confidence_text = (
                f"### {status_emoji} Detection Result\n\n"
                f"**Best Match Similarity:** {best_similarity:.2%}\n\n"
                f"**Threshold:** {config.app.similarity_threshold:.2%}\n\n"
                f"**Detector Used:** {detector_type}\n\n"
                f"**Best Match:** {best_match['name']}\n\n"
                f"**Distance:** {best_match['distance']:.4f}"
            )

            # Prepare gallery images
            gallery_images = []
            for idx, result in enumerate(results):
                # Load ORIGINAL downloaded image (not cropped/aligned)
                stored_path = result.get('original_path') or result['image_path']
                img_path = resolve_image_path(stored_path)

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Add red overlay if similarity exceeds threshold (duplicate detected)
                        if result['similarity'] >= config.app.similarity_threshold:
                            overlay = np.zeros_like(img_rgb)
                            overlay[:, :] = [255, 0, 0]  # Red color in RGB
                            img_rgb = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)  # 30% red opacity

                        # Create caption
                        caption = f"#{idx+1}: {result['name']}, Similarity: {result['similarity']:.2%}"

                        gallery_images.append((img_rgb, caption))
                    else:
                        logger.warning(f"Failed to read image: {img_path}")
                else:
                    logger.warning(f"Image not found: {img_path} (stored as: {stored_path})")

            return status, confidence_text, gallery_images

        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, "", []

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""

        with gr.Blocks(
            title="Face Matching MVP",
            theme=gr.themes.Soft(),
            css="""
                .duplicate-detected { color: red; font-weight: bold; }
                .no-duplicate { color: green; font-weight: bold; }

                /* Gallery caption styling */
                .gallery-item figcaption {
                    text-align: center !important;
                    font-size: 0.75rem !important;
                    line-height: 1.2 !important;
                    padding: 4px !important;
                    white-space: normal !important;
                    word-wrap: break-word !important;
                    overflow: visible !important;
                }
                .gr-gallery .caption {
                    text-align: center !important;
                    font-size: 0.75rem !important;
                    white-space: normal !important;
                }
                .gr-gallery figcaption {
                    text-align: center !important;
                    font-size: 0.75rem !important;
                    padding: 4px 2px !important;
                    line-height: 1.3 !important;
                }
                [data-testid="gallery"] figcaption {
                    text-align: center !important;
                    font-size: 0.75rem !important;
                }

                /* Make gallery wider */
                .wide-gallery {
                    min-width: 100% !important;
                    width: 100% !important;
                }
            """
        ) as demo:
            gr.Markdown(
                """
                # 🔍 Face Matching MVP - Duplicate Account Detection

                Upload a selfie to check if it matches any existing accounts in the database.
                The system uses MobileFaceNet embeddings and Milvus vector search.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📤 Upload Image")
                    input_image = gr.Image(
                        label="Upload Selfie",
                        type="numpy",
                        height=300
                    )

                    detector_choice = gr.Radio(
                        choices=["RetinaFace", "Haar Cascade"],
                        value="RetinaFace",
                        label="Face Detector",
                        info="RetinaFace is more accurate, Haar Cascade is faster"
                    )

                    submit_btn = gr.Button(
                        "🔍 Search for Matches",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=3):
                    # Detection Result section (moved from left column)
                    gr.Markdown("### 📊 Detection Result")
                    status_output = gr.Textbox(
                        label="Status",
                        lines=1,
                        max_lines=1,
                        interactive=False
                    )

                    confidence_output = gr.Markdown()

                    # Gallery output (moved below detection result)
                    gr.Markdown("### 🖼️ Top Similar Faces")
                    gallery_output = gr.Gallery(
                        label="Matches",
                        columns=3,
                        object_fit="contain",
                        show_label=False,
                        allow_preview=True,
                        height="auto",
                        container=True,
                        elem_classes="wide-gallery"
                    )

            # Instructions section
            with gr.Accordion("💡 Instructions", open=False):
                gr.Markdown(
                    """
                    ## How to Use

                    1. **Upload** a clear selfie image containing a face
                    2. **Choose** a face detector:
                       - **RetinaFace**: More accurate, better for challenging images
                       - **Haar Cascade**: Faster, good for well-lit frontal faces
                    3. **Click** "Search for Matches" to find similar faces
                    4. **Review** the results:
                       - **Similarity > 50%**: Duplicate detected ⚠️
                       - **Similarity ≤ 50%**: No duplicate found ✅

                    ## Configuration

                    - **Threshold**: {:.0%} (configured in src/config.py)
                    - **Top K Results**: {} faces
                    - **Embedding Model**: MobileFaceNet (128-dim)
                    - **Vector Database**: Milvus Lite
                    """.format(config.app.similarity_threshold, config.app.top_k_results)
                )

            # Database info
            if self.database_ready:
                stats = self.database.get_stats()
                gr.Markdown(f"**Database Status:** ✅ Ready ({stats['num_entities']} faces indexed)")
            else:
                gr.Markdown(
                    "**Database Status:** ⚠️ Not initialized\n\n"
                    "Run `python prepare_database.py` to populate the database"
                )

            # Connect submit button
            submit_btn.click(
                fn=self.process_image,
                inputs=[input_image, detector_choice],
                outputs=[status_output, confidence_output, gallery_output]
            )

        return demo


def main():
    """Main function to launch the application"""
    logger.info("="*60)
    logger.info("Face Matching MVP - Gradio Application")
    logger.info("="*60)

    # Initialize application
    app = FaceMatchingApp()

    # Create and launch interface
    demo = app.create_interface()

    logger.info("\n" + "="*60)
    logger.info("Launching Gradio interface...")
    logger.info(f"Server: {config.app.host}:{config.app.port}")
    logger.info("="*60 + "\n")

    demo.launch(
        server_name=config.app.host,
        server_port=config.app.port,
        share=config.app.share
    )


if __name__ == "__main__":
    main()
