// Form validation service for Face Sorter web UI

export const validationService = {
  // Directory path validation
  isValidDirectory(path) {
    if (!path || path.trim() === "") {
      return { valid: false, message: "Directory path is required" };
    }

    // Check for invalid characters
    const invalidChars = /[<>:"|?*]/;
    if (invalidChars.test(path)) {
      return {
        valid: false,
        message: "Directory path contains invalid characters",
      };
    }

    // Check for reasonable path length
    if (path.length > 500) {
      return {
        valid: false,
        message: "Directory path is too long",
      };
    }

    return { valid: true };
  },

  // Number validation with range
  isValidNumber(value, min = 1, max = Number.MAX_SAFE_INTEGER) {
    if (value === "" || value === null || value === undefined) {
      return { valid: false, message: "Value is required" };
    }

    const numValue = Number(value);

    if (isNaN(numValue)) {
      return { valid: false, message: "Must be a valid number" };
    }

    if (numValue < min) {
      return { valid: false, message: `Must be at least ${min}` };
    }

    if (numValue > max) {
      return { valid: false, message: `Must be at most ${max}` };
    }

    return { valid: true };
  },

  // Class name validation
  isValidClassName(name) {
    if (!name || name.trim() === "") {
      return { valid: false, message: "Class name is required" };
    }

    if (name.length < 2) {
      return {
        valid: false,
        message: "Class name must be at least 2 characters",
      };
    }

    if (name.length > 100) {
      return {
        valid: false,
        message: "Class name must be less than 100 characters",
      };
    }

    // Check for valid characters
    const validPattern = /^[a-zA-Z0-9_\s-]+$/;
    if (!validPattern.test(name)) {
      return {
        valid: false,
        message:
          "Class name can only contain letters, numbers, underscores, and hyphens",
      };
    }

    return { valid: true };
  },

  // Image quality validation
  isValidQuality(value) {
    const validation = this.isValidNumber(value, 1, 100);
    if (!validation.valid) {
      return validation;
    }

    return { valid: true };
  },

  // Batch size validation
  isValidBatchSize(value) {
    const validation = this.isValidNumber(value, 1, 1000);
    if (!validation.valid) {
      return validation;
    }

    return { valid: true };
  },

  // Form field validation
  validateField(field, value, rules = {}) {
    const results = [];

    if (rules.required && (!value || value === "")) {
      results.push("This field is required");
    }

    if (rules.minLength && value.length < rules.minLength) {
      results.push(`Must be at least ${rules.minLength} characters`);
    }

    if (rules.maxLength && value.length > rules.maxLength) {
      results.push(`Must be less than ${rules.maxLength} characters`);
    }

    if (rules.pattern && !rules.pattern.test(value)) {
      results.push(rules.errorMessage || "Invalid format");
    }

    if (rules.min && Number(value) < rules.min) {
      results.push(`Must be at least ${rules.min}`);
    }

    if (rules.max && Number(value) > rules.max) {
      results.push(`Must be at most ${rules.max}`);
    }

    return {
      valid: results.length === 0,
      errors: results,
    };
  },

  // Complete form validation
  validateForm(formData, validationRules) {
    const errors = {};
    let isValid = true;

    for (const [field, rules] of Object.entries(validationRules)) {
      const value = formData[field];
      const result = this.validateField(field, value, rules);

      if (!result.valid) {
        errors[field] = result.errors;
        isValid = false;
      }
    }

    return {
      valid: isValid,
      errors: errors,
    };
  },

  // Get error message for display
  getErrorMessage(field, errors) {
    if (!errors || !errors[field] || errors[field].length === 0) {
      return "";
    }
    return errors[field][0];
  },

  // Check if form has any errors
  hasErrors(errors) {
    return Object.values(errors).some((fieldErrors) => fieldErrors && fieldErrors.length > 0);
  },
};

export default validationService;