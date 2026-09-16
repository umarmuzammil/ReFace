import os


class ProjectUtils:
    """Project directory manager."""

    def __init__(self, file_path):
        self.file_path = file_path

    def get_file_path(self):
        return self.file_path

    def get_base_name(self):
        return os.path.splitext(os.path.basename(self.file_path))[0]

    def get_base_dir(self):
        return os.path.dirname(self.file_path)

    def create_project_dir(self):
        base = os.path.dirname(self.file_path)
        name = os.path.splitext(os.path.basename(self.file_path))[0]
        project = os.path.join(base, name + "_project")
        for sub in ["extracted_frames", "marked_faces", "matched_frames", "detected_people", "sharp_frames"]:
            os.makedirs(os.path.join(project, sub), exist_ok=True)
        cfg = os.path.join(project, name + "_config.ini")
        if not os.path.exists(cfg):
            with open(cfg, "w") as f:
                f.write(f"video_path={self.file_path}\n")
        return project

    def get_project_dir(self):
        base = os.path.dirname(self.file_path)
        name = os.path.splitext(os.path.basename(self.file_path))[0]
        return os.path.join(base, name + "_project")

    def get_frames_dir(self):
        return os.path.join(self.get_project_dir(), "extracted_frames")

    def get_promoted_frames_dir(self):
        return os.path.join(self.get_project_dir(), "marked_faces")

    def get_detected_faces_dir(self):
        return os.path.join(self.get_project_dir(), "marked_faces")

    def get_saved_frames_dir(self):
        return os.path.join(self.get_project_dir(), "matched_frames")

    def get_detected_figures_dir(self):
        return os.path.join(self.get_project_dir(), "detected_people")

    def get_output_dir(self):
        return os.path.join(self.get_project_dir(), "sharp_frames")

    def get_config_path(self):
        name = os.path.splitext(os.path.basename(self.file_path))[0]
        return os.path.join(self.get_project_dir(), name + "_config.ini")
