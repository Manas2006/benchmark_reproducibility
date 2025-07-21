import React from 'react';

const ListInput = ({
    label,
    value,
    onChange,
    type = "text",
    placeholder = "",
    error = null,
    helperText = "",
    required = false
}) => {
    const handleChange = (e) => {
        // Don't filter empty lines here - let users create new lines with Enter
        const lines = e.target.value.split('\n');
        onChange(lines);
    };

    const displayValue = Array.isArray(value) ? value.join('\n') : '';

    return (
        <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
                {label} {required && <span className="text-red-500">*</span>}
            </label>
            <textarea
                value={displayValue}
                onChange={handleChange}
                rows={4}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${error ? 'border-red-300' : 'border-gray-300'
                    }`}
                placeholder={placeholder}
            />
            {error && (
                <p className="mt-1 text-sm text-red-600">{error}</p>
            )}
            {helperText && (
                <p className="mt-1 text-xs text-gray-500">{helperText}</p>
            )}
        </div>
    );
};

export default ListInput; 