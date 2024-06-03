from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, token="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", quantization_config=quantization_config)
model = AutoModelForCausalLM.from_pretrained(model_id, token="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", quantization_config=quantization_config)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
)


#prompt = pipeline.tokenizer.apply_chat_template(
#        messages, 
#        tokenize=False, 
#        add_generation_prompt=True
#)

input = """
# Captions with timestamps:
00:00:00.000 --> 00:00:03.440
- Welcome back to information visualization and visual analytics.

00:00:04.460 --> 00:00:09.780
- Last week we talked about four general ways of representing trees and hierarchies visually,

00:00:10.460 --> 00:00:15.020
- with emphasis on tree maps and algorithms or procedures to produce them.

00:00:16.840 --> 00:00:22.480
- Today we are going to talk about networks and graphs in terms of more general data structures

00:00:23.000 --> 00:00:25.220
- and of ways of representing them.

00:00:27.560 --> 00:00:36.420
- Graphs and network structures are used very often and we can see them in visual representations in daily life for various purposes,

00:00:36.920 --> 00:00:44.440
- be it as part of floor plans or construction plans or in the form of subway networks or circuit diagrams.

00:00:46.160 --> 00:00:51.960
- There are plenty of domains and tasks where such visual network representations can be helpful.

00:00:56.020 --> 00:01:01.660
- Family relations can be shown as graphs as can be seen here with the relations in game of thrones.

00:01:02.040 --> 00:01:07.220
- You might want to skip this now if you have not seen all seasons and still plan to do so.

00:01:08.160 --> 00:01:14.620
- Initially I wanted to show this in the lecture on trees, since many genealogies have a principal tree layout.

00:01:15.420 --> 00:01:21.360
- But this one here contains also marriages and parent child relations forming a graph.

00:01:21.360 --> 00:01:27.260
- Here you can also already get an idea that networks can show many different relations at once.

00:01:27.840 --> 00:01:30.960
- Here obviously family relations but different ones.

00:01:32.720 --> 00:01:40.660
- And still of course there is a certain timeline integrated here from top to bottom, at least in terms of different generations.

00:01:41.740 --> 00:01:46.060
- The way it is organized helps to identify families in addition.

00:01:47.000 --> 00:01:54.380
- So the layout of a network can obviously convey additional structure as well, or at least support better understanding.

00:01:57.380 --> 00:02:02.740
- Friends networks including those we have now in social media are an additional example.

00:02:03.480 --> 00:02:08.060
- In their entirety they can be huge if you think about the bigger social networks.

00:02:08.920 --> 00:02:16.280
- And new forms of visualization and previous analysis steps need to be made in order to visualize them and make sense of them.

00:02:17.320 --> 00:02:19.980
- Here is a rather old image from Facebook.

00:02:20.440 --> 00:02:26.360
- This is from 2010 already and of course it is probably not meant for analysis.

00:02:27.820 --> 00:02:32.080
- What makes it interesting is that only positions and links are drawn.

00:02:33.260 --> 00:02:39.360
- From this the regions in the world become visible where many persons are using the social network.

00:02:40.360 --> 00:02:55.760
- And this again correlates indirectly also to population density, but obviously also other aspects such as where the social network is popular and whether the population has internet access etc. also plays an important role.

00:02:56.660 --> 00:03:00.060
- So there are many things that can be learned from such a visualization.

00:03:04.220 --> 00:03:16.140
- By the way we are carrying out social media analytics at the Institute of Visualization and Interactive Systems as well and I will present the one or the other approach during the lecture where it fits in.

00:03:17.420 --> 00:03:21.300
- This here is what I showed you already at the beginning of the lecture.

00:03:21.760 --> 00:03:30.240
- Here tweets are shown and the locations the same persons tweeted from are connected indicating that the person has traveled from one place to another.

00:03:31.600 --> 00:03:42.700
- Obviously this also creates a graph and it can provide insights for example on main travel directions and routes which can be interesting for a number of analyses.

00:03:45.040 --> 00:03:52.960
- In the cases I have shown here the layout of the graph is fixed since we would like to note to reflect the respective geoposition.

00:03:53.960 --> 00:03:59.520
- But as you will see later on there are even options to improve graph layout in such cases.

00:04:00.680 --> 00:04:08.360
- Again shapes of countries are not shown explicitly at all. They just emerge here to a certain extent because there is enough data.

00:04:09.460 --> 00:04:16.320
- As mentioned in the introduction lecture the organization of countries in terms of infrastructure becomes visible as well.

00:04:22.100 --> 00:04:35.520
- I would assume that almost all of you have worked with version control systems already and if you did so you might be familiar with revision graphs as they are available from version control systems such as Git.

00:04:36.700 --> 00:04:47.640
- The links here do have a strong temporal or sequential meaning and notes often represent interactions such as commits, merges, foxes and so on.

00:04:49.480 --> 00:05:05.500
- While all this information could also be represented in a textual or symbolic way, many programmers see quite a benefit in having such a graphical representation to understand the distributed collaborative as in chronos development process of coding.

00:05:07.040 --> 00:05:15.060
- And of course the way the graph is shown directly supports the tasks of understanding the just mentioned processes and steps.

00:05:16.860 --> 00:05:23.660
- What you can easily see here is that notes are not shown on the same level, which of course has a reason.

00:05:24.740 --> 00:05:30.760
- The notes themselves need a description in order to make the step they represent comprehensible.

00:05:30.760 --> 00:05:40.180
- And because these descriptions are often not only short labels, they are placed on the same y position next to the shown visual representation.

00:05:41.620 --> 00:05:47.240
- This leads to a trade-off between compactness of the graph and comprehensibility.

00:05:48.500 --> 00:05:58.280
- What I would like to point out when explaining these details is the importance of considering task adequacy when creating visual representations.

00:05:59.560 --> 00:06:06.680
- In the following we will discuss tasks on a very abstract level, but not in a domain-specific way.

00:06:08.080 --> 00:06:19.320
- I will present some general approaches that can be applied in many situations, but please be aware that they might not be the best solution for very specific tasks and usage scenarios.

00:06:23.460 --> 00:06:27.280
- There are also graphs that represent workflows in a visual way.

00:06:28.280 --> 00:06:38.180
- These can convey at least some simple form of logic in terms of defining alternative subsequent steps if a certain condition is met or not.

00:06:39.300 --> 00:06:47.980
- Again, many of you should be familiar with such diagrams since they are often used to define workflows or processes in the context of computer science.

00:06:49.320 --> 00:07:10.300
- Be it in the form of flowcharts, as shown here with this humoristic XKCD example, in the form of state diagrams or activity diagrams, as they are formulated in the unified modeling language, or on a more abstract level for finite state machines, patreon ads and so on.

00:07:11.800 --> 00:07:20.380
- Such diagrams need to be organized and described very thoroughly in terms of labels otherwise they cannot be understood any more easily.

00:07:21.140 --> 00:07:27.180
- You can find plenty more examples where very domain-specific network visualizations are used.

00:07:30.520 --> 00:07:35.980
- Let's again take a quick look at the definition of graphs in terms of the data structure.

00:07:36.800 --> 00:07:44.660
- Many of you have seen this definition already, but some might have not and it certainly does not hurt to repeat that again.

00:07:47.160 --> 00:07:57.980
- A graph is defined by a set of vertices V, also called notes, which can be connected by a set of edges E, which we also call links.

00:07:59.500 --> 00:08:08.840
- More formally, this results for a graph in an ordered pair of vertices and edges and in the case of a directed graph.

00:08:09.280 --> 00:08:21.940
- The edges can be described as a subset of ordered pairs of notes U and V, whereby U and V are part of the set of notes and U is not equal to V.

00:08:22.660 --> 00:08:33.820
- Or if we have the undirected case, then E is a subset of unordered pairs of notes U and V, where the same constraints as above apply.

00:08:35.480 --> 00:08:42.700
- On the right you can see a visual note link representation of a directed graph to illustrate what is formalized here.

00:08:42.700 --> 00:09:00.160
- We have our graph, which contains a set of five notes. These are labeled from A to E and a set of seven directed edges in a form of ordered pairs of vertices with an H A to B and an H B to D, for example.

00:09:02.380 --> 00:09:13.020
- I think this example already makes clear why you would like to actually see a graph in many situations, since it makes it much easier to understand what is encoded by the formalism.

00:09:16.380 --> 00:09:24.600
- Graphs of course can have other important properties. Graphs can have cycles, as opposed to trees, which we discussed in the last lecture.

00:09:26.420 --> 00:09:34.440
- Edges can be associated with weights or additional attributes. A graph can be directed or undirected, as we just saw.

00:09:34.980 --> 00:09:48.520
- Directed graphs are also called digraphs, in short. The number of vertices V is called the order of the graph, and the number of edges E is called the size of the graph.

00:09:49.480 --> 00:09:57.780
- The number of connected edges over vertex is called the degree of the vertex, in degree and out degree for directed graphs.

00:09:58.520 --> 00:10:12.860
- Loops are grounded twice for loop graphs. Undirected graphs are called simple, if they have no multi-egges or loops. If not highlighted otherwise, we will usually assume that we talk about simple graphs.

00:10:15.200 --> 00:10:20.780
- The maximum node degree of a simple graph of order N is N minus 1.

00:10:22.900 --> 00:10:31.600
- What I should mention in addition, multi-graph edges can have the same pair of endpoints, which is not the case for simple graphs.

00:10:32.840 --> 00:10:41.300
- Sometimes loops, as can be seen from the right graph shown here, are also explicitly allowed as part of so-called loop graphs.

00:10:45.760 --> 00:10:52.000
- Unfortunately, the term graph is ambiguous, and refers to a number of concepts in different disciplines.

00:10:52.780 --> 00:10:57.040
- It is even ambiguously used within the field of information visualization.

# Summary: 
"""

input_ids = tokenizer(input, return_tensors="pt").input_ids.to("cuda")

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(input_ids, do_sample=True)

result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(result[0])