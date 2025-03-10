﻿using System.Diagnostics;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;

var sessionOptions = new SessionOptions
{
    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
};

// sessionOptions.AppendExecutionProvider_Tensorrt();
// sessionOptions.AppendExecutionProvider_DML();

sessionOptions.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM);

// sessionOptions.AppendExecutionProvider("CoreML", new() {
//     ["ModelFormat"] = "MLProgram",
//     ["MLComputeUnits"] = "ALL",
// });

sessionOptions.AppendExecutionProvider_CPU();

using var session = new InferenceSession("matmul.onnx", sessionOptions);
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
var t = Tensor.CreateUninitialized<float>([1500, 1500]);
for (var i = 0; i < 1500; i++)
for (var j = 0; j < 1500; j++)
    t[i, j] = Random.Shared.NextSingle() / 2;

for (var i = 0; i < 10; i++)
{
    var x = OrtValue.CreateTensorValueFromSystemNumericsTensorObject(t);
    var s = TimeProvider.System.GetTimestamp();

    using var runOptions = new RunOptions();
    using var outputs = session.Run(
        runOptions,
        new Dictionary<string, OrtValue> { ["X"] = x },
        session.OutputNames
    );
    var y = outputs[0].GetTensorDataAsTensorSpan<float>();
    Console.WriteLine(TimeProvider.System.GetElapsedTime(s).TotalMilliseconds + " ms");
    // t = y.ToDenseTensor();
}
Debugger.Break();
#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
