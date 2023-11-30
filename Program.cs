using System;
using BERTTokenizers;
using BERTTokenizers.Base;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
namespace BertConsole; // Note: actual namespace depends on the project name.

class Program
{
    static void Main(string[] args)
    {
        var runOptions = new RunOptions();
        //using var sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);
        var session = new InferenceSession("./model/distilbert-base-uncased-finetuned-sst-2-english.onnx");

        //var sentence = "You are so handsome";
        var tokenizer = new DistilbertUncasedTokenizer();

        if (args.Length == 0)
        {
            Console.WriteLine("Please input a sentence as argument");
            return;
        }

        foreach (var arg in args)
        {

            
            var tokens = tokenizer.Encode(tokenizer.Tokenize(arg).Count(), arg);
            //Console.WriteLine(string.Join(", ", tokens.Select(x => x.AttentionMask)));
            var bertInput = new BertInput()
            {
                InputIds = tokens.Select(x => x.InputIds).ToArray(),
                InputMask = tokens.Select(x => x.AttentionMask).ToArray()
            };


            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds, new long[] { 1, bertInput.InputIds.Length });
            using var inputMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputMask, new long[] { 1, bertInput.InputMask.Length });
            var inputs = new Dictionary<string, OrtValue> {
            { "input_ids", inputIdsOrtValue },
            { "input_mask", inputMaskOrtValue }
        };

            using var outputs = session.Run(runOptions, inputs, session.OutputNames);

            //Console.WriteLine(string.Join(", ",outputs[0].GetTensorDataAsSpan<float>().ToArray()));
            Console.WriteLine("Text: "+arg);
            var sentiment = getArgmax(outputs[0].GetTensorDataAsSpan<float>());
            Console.WriteLine("Sentiment: " + (Sentiment)sentiment);
        }
    }

    static int getArgmax(ReadOnlySpan<float> span)
    {
        var max = span[0];
        var maxIndex = 0;
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > max)
            {
                max = span[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    enum Sentiment
    {
        Negative = 0,
        Positive = 1
    }
}


public class DistilbertUncasedTokenizer : UncasedTokenizer
{
    public DistilbertUncasedTokenizer() : base("./vocab.txt")
    {
    }
}
public struct BertInput
{
    public long[] InputIds { get; set; }
    public long[] InputMask { get; set; }
}
