/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useState } from "react";
import { useML } from "@/context/MLContext";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  FileDown,
  Wand2,
  SlidersHorizontal,
  Tag,
  Info,
  Check,
  X,
} from "lucide-react";
import { preprocessData, getDownloadPreprocessedUrl } from "@/services/api";
import { motion } from "motion/react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Checkbox } from "@/components/ui/checkbox";

export function PreprocessStep() {
  const {
    files,
    summaries,
    missingStrategy,
    setMissingStrategy,
    scaling,
    setScaling,
    encoding,
    setEncoding,
    targetColumn,
    setTargetColumn,
    suggestedMissingStrategies,
    suggestedTargetColumns,
    setActiveStep,
    setError,
    isLoading,
    setIsLoading,
    setProgress,
    setPreprocessedFiles,
  } = useML();

  // New state variables for column selection
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [scalingColumns, setScalingColumns] = useState<string[]>([]);
  const [encodingColumns, setEncodingColumns] = useState<string[]>([]);
  const [showScalingColumnSelector, setShowScalingColumnSelector] =
    useState(false);
  const [showEncodingColumnSelector, setShowEncodingColumnSelector] =
    useState(false);

  useEffect(() => {
    if (Object.keys(suggestedMissingStrategies).length) {
      setMissingStrategy(
        Object.values(suggestedMissingStrategies)[0] as string
      );
    }

    // Set default target column for target encoding
    if (Object.keys(suggestedTargetColumns).length) {
      const target = Object.values(suggestedTargetColumns)[0];
      if (target) {
        setTargetColumn(target);
      }
    }

    // Extract all unique column names from all files
    if (Object.keys(summaries).length > 0) {
      try {
        // Safely get all columns
        const allColumns = Object.values(summaries)
          .filter(
            (s: any) => s && s.summary && Array.isArray(s.summary.columns)
          )
          .flatMap((s: any) => s.summary.columns)
          .filter((v: any, i: any, a: any) => a.indexOf(v) === i);

        setAvailableColumns(allColumns);

        // Safely identify numeric columns
        const numericColumns = Object.values(summaries)
          .filter((s: any) => s && s.summary && s.summary.dtypes)
          .flatMap((s: any) =>
            s.summary.columns.filter((col: string) => {
              const dtype = s.summary.dtypes[col];
              return (
                dtype === "int64" ||
                dtype === "float64" ||
                dtype === "int" ||
                dtype === "float" ||
                dtype?.includes("float") ||
                dtype?.includes("int")
              );
            })
          )
          .filter((v: any, i: any, a: any) => a.indexOf(v) === i);

        setScalingColumns(numericColumns);

        // Safely identify categorical columns
        const categoricalColumns = Object.values(summaries)
          .filter((s: any) => s && s.summary && s.summary.dtypes)
          .flatMap((s: any) =>
            s.summary.columns.filter((col: string) => {
              const dtype = s.summary.dtypes[col];
              return (
                dtype === "object" ||
                dtype === "category" ||
                dtype?.includes("str") ||
                dtype?.includes("O")
              );
            })
          )
          .filter((v: any, i: any, a: any) => a.indexOf(v) === i);

        setEncodingColumns(categoricalColumns);
      } catch (error) {
        console.error("Error processing column data:", error);
        // Set empty defaults if there's an error
        setAvailableColumns([]);
        setScalingColumns([]);
        setEncodingColumns([]);
      }
    }
  }, [
    suggestedMissingStrategies,
    suggestedTargetColumns,
    setMissingStrategy,
    setTargetColumn,
    summaries,
  ]);

  useEffect(() => {
    if (summaries && Object.keys(summaries).length > 0) {
      const allColumns = Object.values(summaries).flatMap(
        (s: any) => s.summary.columns
      );
      const uniqueColumns = Array.from(new Set(allColumns));
      const categoricalCols = uniqueColumns.filter((col) =>
        Object.values(summaries).some(
          (s: any) => s.summary.data_types[col] === "object"
        )
      );
      const numericCols = uniqueColumns.filter((col) =>
        Object.values(summaries).some(
          (s: any) =>
            s.summary.data_types[col] === "float64" ||
            s.summary.data_types[col] === "int64"
        )
      );
      setEncodingColumns(categoricalCols);
      setScalingColumns(numericCols);
    }
  }, [summaries]);

  const handlePreprocess = async () => {
    if (!files.length) return setError("Please upload files first.");

    // Validate target column is selected for target-based encoding methods
    if ((encoding === "target" || encoding === "kfold") && !targetColumn) {
      return setError("Please select a target column for target encoding.");
    }

    // Validate that some columns are selected for scaling if scaling is enabled
    if (scaling && showScalingColumnSelector && scalingColumns.length === 0) {
      return setError(
        "Please select at least one column for scaling or disable scaling."
      );
    }

    // Validate that some columns are selected for encoding
    if (showEncodingColumnSelector && encodingColumns.length === 0) {
      return setError("Please select at least one column for encoding.");
    }

    setIsLoading(true);
    setProgress(10);

    try {
      const data = await preprocessData(
        files,
        missingStrategy,
        scaling,
        showScalingColumnSelector ? scalingColumns.join(",") : "",
        encoding,
        showEncodingColumnSelector ? encodingColumns.join(",") : "",
        targetColumn, // Always pass targetColumn, not just for target encoding
        setProgress
      );
      setPreprocessedFiles(
        Object.fromEntries(
          Object.entries(data).map(([k, v]: [string, any]) => [
            k,
            v.preprocessed_file,
          ])
        )
      );
      setActiveStep("train");
    } catch (error: any) {
      setError(
        `Preprocessing failed: ${error.response?.data?.error || error.message}.`
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPreprocessed = (filename: string) => {
    window.location.href = getDownloadPreprocessedUrl(filename);
  };

  // Check if encoding requires a target column
  const isTargetEncodingMethod = encoding === "target" || encoding === "kfold";

  // Toggle column selection for scaling
  const toggleScalingColumn = (column: string) => {
    if (scalingColumns.includes(column)) {
      setScalingColumns(scalingColumns.filter((col) => col !== column));
    } else {
      setScalingColumns([...scalingColumns, column]);
    }
  };

  // Toggle column selection for encoding
  const toggleEncodingColumn = (column: string) => {
    if (encodingColumns.includes(column)) {
      setEncodingColumns(encodingColumns.filter((col) => col !== column));
    } else {
      setEncodingColumns([...encodingColumns, column]);
    }
  };

  // Helper to determine if a column is numeric - safely
  const isNumericColumn = (column: string) => {
    try {
      return Object.values(summaries).some(
        (s: any) =>
          s &&
          s.summary &&
          s.summary.columns &&
          s.summary.columns.includes(column) &&
          s.summary.dtypes &&
          (s.summary.dtypes[column] === "int64" ||
            s.summary.dtypes[column] === "float64" ||
            s.summary.dtypes[column] === "int" ||
            s.summary.dtypes[column] === "float" ||
            String(s.summary.dtypes[column]).includes("int") ||
            String(s.summary.dtypes[column]).includes("float"))
      );
    } catch (error) {
      console.error("Error checking numeric column:", error);
      return false;
    }
  };

  // Helper to determine if a column is categorical - safely
  const isCategoricalColumn = (column: string) => {
    try {
      return Object.values(summaries).some(
        (s: any) =>
          s &&
          s.summary &&
          s.summary.columns &&
          s.summary.columns.includes(column) &&
          s.summary.dtypes &&
          (s.summary.dtypes[column] === "object" ||
            s.summary.dtypes[column] === "category" ||
            String(s.summary.dtypes[column]).includes("str") ||
            String(s.summary.dtypes[column]).includes("O"))
      );
    } catch (error) {
      console.error("Error checking categorical column:", error);
      return false;
    }
  };

  if (Object.keys(summaries).length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="border border-white/10 bg-secondary-50 shadow-md overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-secondary-100 to-secondary-50 border-b border-white/10">
          <CardTitle className="text-2xl text-white">Preprocess Data</CardTitle>
          <CardDescription className="text-white/70">
            Configure preprocessing options for your datasets
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6 space-y-8">
          <Alert className="bg-primary/10 border-primary/20">
            <Info className="h-4 w-4 text-primary" />
            <AlertDescription className="text-white">
              Our AI has analyzed your data and suggested optimal preprocessing
              settings. You can adjust these settings if needed.
            </AlertDescription>
          </Alert>

          <div className="grid md:grid-cols-2 gap-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-secondary-100 p-5 rounded-lg border border-white/10 shadow-sm"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-primary/20 rounded-full flex items-center justify-center">
                  <Wand2 className="h-5 w-5 text-primary" />
                </div>
                <h3 className="font-medium text-white">
                  Missing Values Strategy
                </h3>
              </div>

              {suggestedMissingStrategies &&
                Object.values(suggestedMissingStrategies).length > 0 && (
                  <div className="mb-3">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/20 text-primary">
                      Suggested:{" "}
                      {Object.values(suggestedMissingStrategies).join(", ")}
                    </span>
                  </div>
                )}

              <select
                value={missingStrategy}
                onChange={(e) => setMissingStrategy(e.target.value)}
                className="w-full p-3 rounded-md border border-white/10 bg-secondary-200 text-white focus:border-primary focus:ring focus:ring-primary/20 transition-all"
                disabled={isLoading}
              >
                <option className="bg-primary text-black" value="mean">
                  Mean
                </option>
                <option className="bg-primary text-black" value="median">
                  Median
                </option>
                <option className="bg-primary text-black" value="mode">
                  Mode
                </option>
                <option className="bg-primary text-black" value="drop">
                  Drop
                </option>
              </select>
              <p className="mt-2 text-xs text-white/50">
                {missingStrategy === "mean" &&
                  "Replace missing values with the mean of the column"}
                {missingStrategy === "median" &&
                  "Replace missing values with the median of the column"}
                {missingStrategy === "mode" &&
                  "Replace missing values with the most frequent value"}
                {missingStrategy === "drop" &&
                  "Remove rows with missing values"}
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-secondary-100 p-5 rounded-lg border border-white/10 shadow-sm"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-primary/20 rounded-full flex items-center justify-center">
                  <SlidersHorizontal className="h-5 w-5 text-primary" />
                </div>
                <h3 className="font-medium text-white">Scaling & Encoding</h3>
              </div>

              <div className="flex items-center mb-4">
                <input
                  type="checkbox"
                  id="scaling"
                  checked={scaling}
                  onChange={(e) => setScaling(e.target.checked)}
                  className="w-4 h-4 text-primary border-white/30 rounded focus:ring-primary bg-secondary-200"
                  disabled={isLoading}
                />
                <label
                  htmlFor="scaling"
                  className="ml-2 text-sm font-medium text-white"
                >
                  Enable Scaling
                </label>
                {scaling && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="ml-2 text-xs text-primary"
                    onClick={() =>
                      setShowScalingColumnSelector(!showScalingColumnSelector)
                    }
                  >
                    {showScalingColumnSelector
                      ? "Hide Column Selector"
                      : "Select Columns"}
                  </Button>
                )}
              </div>
              <p className="text-xs text-white/50 mb-4">
                Standardize numeric features to have zero mean and unit variance
              </p>

              <div className="mb-2">
                <label className="block text-sm font-medium text-white mb-1">
                  Encoding Method
                </label>
                <div className="flex items-center">
                  <select
                    value={encoding}
                    onChange={(e) => setEncoding(e.target.value)}
                    className="w-full p-3 rounded-md border border-white/10 bg-secondary-200 text-white focus:border-primary focus:ring focus:ring-primary/20 transition-all"
                    disabled={isLoading}
                  >
                    <option className="bg-primary text-black" value="onehot">
                      One-Hot Encoding
                    </option>
                    <option className="bg-primary text-black" value="label">
                      Label Encoding
                    </option>
                    <option className="bg-primary text-black" value="target">
                      Target Encoding
                    </option>
                    <option className="bg-primary text-black" value="kfold">
                      K-Fold Target Encoding
                    </option>
                  </select>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="ml-2 text-xs text-primary whitespace-nowrap"
                    onClick={() =>
                      setShowEncodingColumnSelector(!showEncodingColumnSelector)
                    }
                  >
                    {showEncodingColumnSelector
                      ? "Hide Columns"
                      : "Select Columns"}
                  </Button>
                </div>
              </div>
              <p className="text-xs text-white/50">
                {encoding === "target" &&
                  "Target encoding uses the target variable to encode categorical features"}
                {encoding === "kfold" &&
                  "K-Fold target encoding prevents data leakage by using cross-validation"}
                {encoding === "label" &&
                  "Label encoding converts categories to numeric values"}
                {encoding === "onehot" &&
                  "One-hot encoding creates binary columns for each category"}
              </p>
            </motion.div>
          </div>

          {/* Column Selector for Scaling */}
          {scaling && showScalingColumnSelector && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-secondary-100 p-5 rounded-lg border border-white/10 shadow-sm"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-white">
                  Select Columns for Scaling
                </h3>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-xs border-primary text-primary"
                    onClick={() =>
                      setScalingColumns(
                        availableColumns.filter((col) => isNumericColumn(col))
                      )
                    }
                  >
                    <Check className="mr-1 h-3 w-3" />
                    Select All Numeric
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-xs border-primary text-primary"
                    onClick={() => setScalingColumns([])}
                  >
                    <X className="mr-1 h-3 w-3" />
                    Clear All
                  </Button>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto p-2">
                {availableColumns.map((column) => (
                  <div
                    key={`scale-${column}`}
                    className={`flex items-center p-2 rounded ${
                      isNumericColumn(column)
                        ? "bg-secondary-200"
                        : "bg-secondary-200/50 text-white/50"
                    }`}
                  >
                    <Checkbox
                      id={`scale-${column}`}
                      checked={scalingColumns.includes(column)}
                      onCheckedChange={() => toggleScalingColumn(column)}
                      disabled={!isNumericColumn(column) || isLoading}
                      className="mr-2"
                    />
                    <label
                      htmlFor={`scale-${column}`}
                      className="text-sm cursor-pointer truncate"
                    >
                      {column}
                    </label>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs text-white/50">
                Only numeric columns can be scaled. Selected:{" "}
                {scalingColumns.length} columns
              </p>
            </motion.div>
          )}

          {/* Column Selector for Encoding */}
          {showEncodingColumnSelector && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-secondary-100 p-5 rounded-lg border border-white/10 shadow-sm"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-white">
                  Select Columns for Encoding
                </h3>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-xs border-primary text-primary"
                    onClick={() =>
                      setEncodingColumns(
                        availableColumns.filter((col) =>
                          isCategoricalColumn(col)
                        )
                      )
                    }
                  >
                    <Check className="mr-1 h-3 w-3" />
                    Select All Categorical
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-xs border-primary text-primary"
                    onClick={() => setEncodingColumns([])}
                  >
                    <X className="mr-1 h-3 w-3" />
                    Clear All
                  </Button>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto p-2">
                {availableColumns.map((column) => (
                  <div
                    key={`encode-${column}`}
                    className={`flex items-center p-2 rounded ${
                      isCategoricalColumn(column)
                        ? "bg-secondary-200"
                        : "bg-secondary-200/50 text-white/50"
                    }`}
                  >
                    <Checkbox
                      id={`encode-${column}`}
                      checked={encodingColumns.includes(column)}
                      onCheckedChange={() => toggleEncodingColumn(column)}
                      disabled={!isCategoricalColumn(column) || isLoading}
                      className="mr-2"
                    />
                    <label
                      htmlFor={`encode-${column}`}
                      className="text-sm cursor-pointer truncate"
                    >
                      {column}
                    </label>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs text-white/50">
                Categorical columns are recommended for encoding. Selected:{" "}
                {encodingColumns.length} columns
              </p>
            </motion.div>
          )}

          {isTargetEncodingMethod && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-secondary-100 p-5 rounded-lg border border-white/10 shadow-sm"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-primary/20 rounded-full flex items-center justify-center">
                  <Tag className="h-5 w-5 text-primary" />
                </div>
                <h3 className="font-medium text-white">
                  Target Column (Required for{" "}
                  {encoding === "target" ? "Target" : "K-Fold"} Encoding)
                </h3>
              </div>

              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full p-3 rounded-md border border-white/10 bg-secondary-200 text-white focus:border-primary focus:ring focus:ring-primary/20 transition-all"
                disabled={isLoading}
              >
                <option value="">Select Target Column</option>
                {availableColumns.map((col: string) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </motion.div>
          )}
        </CardContent>
        <CardFooter className="px-6 py-4 bg-secondary-200 border-t border-white/10 flex flex-col gap-4">
          <Button
            onClick={handlePreprocess}
            className="w-full bg-primary hover:bg-secondary/90 hover:text-white text-black font-semibold h-12 text-base border-2"
            disabled={isLoading}
          >
            {isLoading ? "Processing..." : "Preprocess Data"}
          </Button>

          {files.map((file: File) => (
            <Button
              key={file.name}
              variant="outline"
              onClick={() => handleDownloadPreprocessed(file.name)}
              className="w-full border-primary text-primary hover:bg-primary/10 hover:text-primary-foreground font-medium"
              disabled={isLoading}
            >
              <FileDown className="mr-2 h-4 w-4" />
              Download Preprocessed {file.name}
            </Button>
          ))}
        </CardFooter>
      </Card>
    </motion.div>
  );
}
