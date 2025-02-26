using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

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

var t = new DenseTensor<float>([1500, 1500]);
for (var i = 0; i < 1500; i++)
for (var j = 0; j < 1500; j++)
    t[i, j] = Random.Shared.NextSingle() / 2;

for (var i = 0; i < 10; i++)
{
    var x = NamedOnnxValue.CreateFromTensor("X", t);
    var s = TimeProvider.System.GetTimestamp();
    using var outputs = session.Run([x]);
    var y = outputs[0].AsTensor<float>();
    Console.WriteLine(TimeProvider.System.GetElapsedTime(s).TotalMilliseconds + " ms");
    // t = y.ToDenseTensor();
}
Debugger.Break();
