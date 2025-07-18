import React from 'react';

const RawSbatchInput = ({ value, onChange, error }) => {
    return (
        <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
                Raw SBATCH Directives
            </label>
            <textarea
                value={value}
                onChange={onChange}
                rows={8}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${error ? 'border-red-300' : 'border-gray-300'
                    }`}
                placeholder="#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1"
            />
            {error && (
                <p className="mt-1 text-sm text-red-600">{error}</p>
            )}
            <p className="mt-1 text-xs text-gray-500">
                Enter raw SBATCH directives. Each line will be included in the generated script.
            </p>
        </div>
    );
};

export default RawSbatchInput; 