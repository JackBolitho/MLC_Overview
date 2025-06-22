/*
Jack Bolitho  March 27 2025

This script is a reconstruction of the MLC LLM threaded engine, which is how the chatbot is implemented without Python wrapping.
This is an example program and is unfinished. Running the program will print out a number of untokenized, sampled outputs. 
These outputs are returned via a callback function RequestStreamCallbackFunc.

When making this project with CMake, use the flag -DUSE_METAL=ON to allow the correct loader to be avaliable.
You can run compile_threaded_engine.sh to compile the program.

NOTE: There is a 32 byte memory leak caused by calling threadedEngine->Reload(). Resetting the engine and unloading it
does not fix the problem. It may be caused by an internal MLC LLM issue.

The commit was January 20 2025. This is the commit of mlc-llm you should use:
https://github.com/mlc-ai/mlc-llm/tree/a175d4420ef2e8e9da2a8c1b237c361a78dc761e

As of June 22 2025, some files in tvm have been moved, so use the commit:
https://github.com/mlc-ai/relax/tree/7ed4584952546fa5d54366b72a6862f919c18daa
when cloning mlc-llm/3rdparty/tvm

*/


#include "mlc-llm/cpp/serve/threaded_engine.h"
#include "mlc-llm/cpp/support/json_parser.h"
#include "mlc-llm/cpp/serve/config.h"

#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>

#include <thread>
#include <set>
#include <stdio.h>

using namespace mlc::llm;
using namespace mlc::llm::serve;
using namespace tvm::runtime;


class ThreadedEngineInterface
{

private:
    std::unique_ptr<ThreadedEngine> threadedEngine;
    std::unique_ptr<std::thread> backgroundThread;
    std::unique_ptr<std::thread> streambackLoopThread;
    std::set<std::string> requestIDs;

    //create the request to pass to the engine, taken from threaded_engine.cc 239
    static Request CreateRequest(std::unique_ptr<ThreadedEngine>& threadedEngine, String id, Array<Data> inputs, String generation_cfg_json_str)
    {
        picojson::object config = json::ParseToJSONObject(generation_cfg_json_str);
        auto gen_config = GenerationConfig::FromJSON(config, threadedEngine->GetDefaultGenerationConfig());
        CHECK(gen_config.IsOk()) << gen_config.UnwrapErr();
        return Request(std::move(id), std::move(inputs), gen_config.Unwrap());
    }

    //packed function for background loop
    static PackedFunc CreateRunBackgroundLoopFunc(std::unique_ptr<ThreadedEngine>& threadedEngine) {
        return PackedFunc([&threadedEngine](TVMArgs args, TVMRetValue* rv) {
            threadedEngine->RunBackgroundLoop();
        });
    }

    //packed function for streamback loop
    static PackedFunc CreateRunSteambackLoopFunc(std::unique_ptr<ThreadedEngine>& threadedEngine) {
        return PackedFunc([&threadedEngine](TVMArgs args, TVMRetValue* rv) {
            threadedEngine->RunBackgroundStreamBackLoop();
        });
    }

    //called when inferenced
    static void RequestStreamCallbackFunc(TVMArgs args, TVMRetValue* rv) 
    {
        if(args.num_args > 0){
            //first, cast the arguments into a tvm array of RequestStreamOutputs
            Array<RequestStreamOutput> requestStreamOutputList = args[0];

            //then get the first output value
            RequestStreamOutput requestStreamOutput = requestStreamOutputList[0];

            //then get the group_delta_token_ids from the requestStreamOutput
            auto groupIDs = requestStreamOutput->group_delta_token_ids;

            //print out first element of the matrix
            std::cout << groupIDs[0][0] << std::endl;

        }else{
            std::cout << "No token outputted" << std::endl; 
        }
    }

public:
    //Construct the threaded engine
    ThreadedEngineInterface(std::string engine_config_json_str, Device device)
    {
        //create threaded engine, corresponds with engine_base.py 605
        threadedEngine = ThreadedEngine::Create();

        //initalize threaded engine, corresponds with engine_base.py 623
        PackedFunc packed_func = PackedFunc(RequestStreamCallbackFunc);
        auto trace_recorder = EventTraceRecorder::Create();
        threadedEngine->InitThreadedEngine(device, packed_func, trace_recorder);

        //start background threads, corresponds with engine_base.py 629, 630
        PackedFunc backgroundLoop = CreateRunBackgroundLoopFunc(threadedEngine);
        backgroundThread = std::make_unique<std::thread>(backgroundLoop);

        PackedFunc backgroundSteambackLoop = CreateRunSteambackLoopFunc(threadedEngine);
        streambackLoopThread = std::make_unique<std::thread>(backgroundSteambackLoop);

        //load in parameters with config as json str, corresponds with engine_base.py 646
        threadedEngine->Reload(engine_config_json_str);
    }

    //Pass requests to the threadedEngine to complete the given token_id input, which is returned via a callback function
    void ChatCompletion(std::vector<int32_t> token_ids, std::string generation_config_json_str, std::string requestID)
    {
        //add to tracking list
        requestIDs.insert(requestID);

        //prepare inputs
        TokenData token_data(token_ids); //cast token_ids to mlc token data
        Array<Data> inputs = {token_data}; //implicit cast to mlc data

        //create a request to send to the engine, corresponds with engine.py 1888, _generate
        Request engineRequest = CreateRequest(threadedEngine, requestID, inputs, generation_config_json_str);

        //add a request to the engine, corresponds with engine.py 1897
        threadedEngine->AddRequest(engineRequest);

        //yield to prevent ExitBackgroundLoop from cutting off loop too early
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    //stop and cleanup engine
    void Stop()
    {
        //exit the background loop and terminate model execution, corresponds with engine_base.py terminate 656
        for(std::string request : requestIDs){
            threadedEngine->AbortRequest(request);
        }
        threadedEngine->ExitBackgroundLoop();

        threadedEngine->Unload();

        //join the threads
        while(!(*backgroundThread).joinable() || !(*streambackLoopThread).joinable()){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        (*backgroundThread).join();
        (*streambackLoopThread).join();
    }
};

int main()
{
    //safety check
    FILE* weights = fopen("JordanAI-bassAndChords-v0.1-MLCLLM", "r");
    if(weights == NULL){
        std::cerr << "Error: The MLC compiled weights are not in the current directory." << std::endl;
        return 1;
    }
    fclose(weights);

    //set up example
    Device device = {kDLMetal, 0};
    std::string engine_config_json_str = "{\"model\": \"JordanAI-bassAndChords-v0.1-MLCLLM\", \"model_lib\": \"JordanAI-bassAndChords-v0.1-MLCLLM/MLCModel.dylib\", \"additional_models\": [], \"mode\": \"local\", \"tensor_parallel_shards\": null, \"pipeline_parallel_stages\": null, \"opt\": null, \"gpu_memory_utilization\": null, \"kv_cache_page_size\": 16, \"max_num_sequence\": null, \"max_total_sequence_length\": null, \"max_single_sequence_length\": null, \"prefill_chunk_size\": null, \"sliding_window_size\": null, \"attention_sink_size\": null, \"max_history_size\": null, \"kv_state_kind\": null, \"speculative_mode\": \"disable\", \"spec_draft_length\": 0, \"spec_tree_width\": 1, \"prefix_cache_mode\": \"radix\", \"prefix_cache_max_num_recycling_seqs\": null, \"prefill_mode\": \"hybrid\", \"verbose\": true}";
    std::string generation_config_json_str = "{\"n\":1,\"temperature\":null,\"top_p\":null,\"frequency_penalty\":null,\"presence_penalty\":null,\"repetition_penalty\":null,\"logprobs\":false,\"top_logprobs\":0,\"logit_bias\":null,\"max_tokens\":20,\"seed\":100,\"stop_strs\":[],\"stop_token_ids\":[2],\"response_format\":null,\"debug_config\":null}";
    std::string requestID = "chatcmpl-11b0e8277bcc4b8b93dcd3e43176918e";
    std::vector<int32_t> token_ids = {55027}; //input

    //inference
    ThreadedEngineInterface threadedEngine(engine_config_json_str, device);
    threadedEngine.ChatCompletion(token_ids, generation_config_json_str, requestID);
    threadedEngine.Stop();

    return 0;
}