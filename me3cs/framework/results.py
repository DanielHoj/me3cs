class Results:
    def __init__(self) -> None:
        self.calibration = None
        self.cross_validation = None
        self.diagnostics = None
        self.optimal_number_component = None

    def __repr__(self):
        if self.diagnostics is not None:
            cal = ", ".join(self.calibration.__dict__.keys())
            cross_validation = ", ".join(self.cross_validation.__dict__.keys())
            diagnostics = ", ".join(self.diagnostics.__dict__.keys())
        else:
            cal, cross_validation, diagnostics = "None", "None", "None"
        return f"me3cs results calculated:\n" \
               f"Calibration: {cal}\n" \
               f"Cross_validation: {cross_validation}\n" \
               f"Diagnostrics: {diagnostics}\n" \
               f"Optimal components: {self.optimal_number_component}"
