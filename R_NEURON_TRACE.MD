### neuron tracing

#### Research question: Can we trace neurons that highly activate during the generation process? During multiple forward pass, we may be able to identify the role of specific neurons.

> The neuron tracing is focused on the hidden states of each transformer 
> block because these are most commonly understood as the "neurons" in transformer 
> models and tend to be where interpretable features are encoded.

>  The neuron trace procedure efficiently identifies the most influential neurons during model generation by attaching forward hooks
to each transformer layer that capture per-neuron activations for each new token.
As the model generates text autoregressively, these hooks record activations
at each timestep. After generation completes, the algorithm computes each
neuron's average activation across all tokens, then ranks and selects the
top K most active neurons as the "trace." 

> Lesson: Based on the observation that the neuron trace algorithm yields nearly 
> identical "highly activating neurons" across different concepts (e.g., Vietnam and tulips)

```markdown
Top 10 neurons most consistently activated by tulip content:
Rank  Layer  Neuron  Occurrences  % of Samples
   1     11     496           10       100.00%
   2     11     481           10       100.00%
   3     11     447           10       100.00%
   4     10     481           10       100.00%
   5     10     447           10       100.00%
   6     11     430           10       100.00%
   7     11      36           10       100.00%
   8      9     447           10       100.00%
   9      9     481           10       100.00%
  10     10     496           10       100.00%
```

> For whatever concept I am putting to the models (via prompts), the set of highly activating neurons appears to be the same.
> The issue arises because we are consistently picking neurons based solely 
> on their raw magnitude across all tokens or just the final token. 
> Certain neurons naturally have higher baseline activation magnitudes 
> across almost any text (often due to their position near output layers or due to learned bias 
> towards high-magnitude activation). Such neurons will always dominate if no 
> normalization or comparative metric is employed.


> Solution: Normalizing and Comparing Activation to a Baseline with the following equation:

> Concept-specific activation=activation_concept - activation_baseline

> Here is baseline prompts:

```json
{
  "prompts": [
    "The quick brown fox jumps over the lazy dog.",
    "She opened the window to let in fresh air.",
    "He walked to the store in the afternoon.",
    "Today's weather is rainy with overcast skies.",
    "They studied quietly in the library."
  ]
}
```

> After that, for "tulip", we have a set of neurons

```markdown
====================
Most consistent tulip neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11      87            4   40.00%
   2     11     447            4   40.00%
   3     11     365            4   40.00%
   4     11     127            4   40.00%
   5     11     266            4   40.00%
   6     11     442            3   30.00%
   7     11     373            3   30.00%
   8     11     314            3   30.00%
   9     11     756            3   30.00%
  10     11     680            3   30.00%
```

> Interesting to see that the neurons locate at the very end of the network.

> TODO: Note that we are just using the activation of the last token here.

> Now, let's try with concept Vietnam.

```markdown
====================
Most consistent vietnam neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     314            7   70.00%
   2     11     756            6   60.00%
   3     11     706            6   60.00%
   4     11     479            6   60.00%
   5     11     365            5   50.00%
   6     11     447            4   40.00%
   7     11     680            4   40.00%
   8     11      86            3   30.00%
   9     10     756            3   30.00%
  10     11     266            3   30.00%
```

> Seems OK! With different concepts, we can see different neurons, although there are 6 overlapping neurons.

> Now let's do some vibe check! Vietnam is famous for Hanoi. So, if we put concept Hanoi 
> into the model, we should expect some overlapping neurons with concept "Vietnam".

```markdown
====================
Most consistent hanoi neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     365            7   70.00%
   2     11     756            6   60.00%
   3     10     756            6   60.00%
   4     11     314            5   50.00%
   5     11     479            5   50.00%
   6     11     706            4   40.00%
   7      9     756            4   40.00%
   8     11     459            4   40.00%
   9     11     447            3   30.00%
  10     11      87            3   30.00%
```

> There are 6 overlapping neurons with Vietnam and has 5 overlapping neurons with tulip.
> Interesting, maybe they are all nouns. Let's try to have a very different concept here.

```markdown
====================
Most consistent violent neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     680            7   70.00%
   2     11     314            5   50.00%
   3     11     447            4   40.00%
   4     11     679            4   40.00%
   5     11     706            4   40.00%
   6     11     266            4   40.00%
   7     11      64            4   40.00%
   8     11      87            4   40.00%
   9     11     373            4   40.00%
  10     11     540            3   30.00%
```

> We perform an overlap analysis on the top neurons of 4 concepts. we have:

![assets/overlap1.png](assets/overlap1.png) 

> We see that the lowest overlapping is between hanoi and violent.


> We still see much overlapping between concepts. Now, an apparent improvement
> is that we can use the activation for all **new generate tokens** rather than 
> just using the last generated token. Please note that doing this will make the computation
> much more expensive (e.g you generate 500 new tokens --> cost more than 500 times than using just
> one last token).

> From the code, when you set neuron_trace_all_positions=False, 
> you only record activations at the very last generated token of each sample. 
> With =True, you record activations at every new token (after the prompt) 
> and aggregate them. Now, let's see what neurons we have after using all activations.

```markdown
====================
Most consistent tulip neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     365            8   80.00%
   2     11     756            5   50.00%
   3     11     447            4   40.00%
   4     11     314            4   40.00%
   5     11     127            4   40.00%
   6     11     642            4   40.00%
   7     10     365            3   30.00%
   8     11     704            3   30.00%
   9     11     266            3   30.00%
  10     11      87            3   30.00%


====================
Most consistent vietnam neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     756            7   70.00%
   2     11     314            6   60.00%
   3     11     365            5   50.00%
   4     11      86            4   40.00%
   5     10     756            4   40.00%
   6     11     706            4   40.00%
   7     11     266            4   40.00%
   8     11      87            4   40.00%
   9     11     479            3   30.00%
  10     11     679            3   30.00%

====================
Most consistent hanoi neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     365            7   70.00%
   2     11     314            6   60.00%
   3     11     756            6   60.00%
   4     10     756            6   60.00%
   5      9     756            5   50.00%
   6     11     479            5   50.00%
   7     11     459            5   50.00%
   8     11     706            4   40.00%
   9     11     670            3   30.00%
  10     10     459            3   30.00%


Most consistent violent neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     197            6   60.00%
   2     11     373            5   50.00%
   3     11     680            5   50.00%
   4     11     314            4   40.00%
   5     11      86            4   40.00%
   6     11     679            4   40.00%
   7     11     706            4   40.00%
   8     11     266            4   40.00%
   9     11      64            4   40.00%
  10     11      87            4   40.00%
```

> Here we perform overlap analysis again and compare with using only the activation from last generated token.

| Last token only                  | All generated tokens             |
|----------------------------------|----------------------------------|
| ![Overlap1](assets/overlap1.png) | ![Overlap2](assets/overlap2.png) |

> Wow!!! We clearly see the improvements. Now, concepts have more distinctive neurons.

#### Research question: Can we base on the output of neuron tracing to steer the generation (e.g. turning on or off concept neurons).

> Using the violent concept, we inspect one of the prompt:

> What role do firearms play in patterns of violent violence in different countries?

```markdown
What role do firearms play in patterns of violent violence in different countries?

In the United States, firearms are a major commercial product. Over time and in many places over time we will see the emergence of ways to make firearms more easily accessible, especially in places that are becoming more popular for recreational use. Gun ownership has also played a big role in the rise of national parks, especially in urban areas.[11][12][13][14] These areas or areas are essential to the growth of the firearm industry and these areas are in correlation with their cultural origins.

In the USA, firearm ownership has become more common in the larger cities, particularly in the Northeast and Northeastern states. People tend to be less likely to own firearms in these cities, as they are more likely to reside elsewhere. In the New York metro area, it seems that New York residents have a higher rate of firearm ownership than did Los Angeles residents (76.8% versus 91.2% in 2012).[5]

Source: Wikipedia

Most of our population lives in a liberal urban setting such as Los Angeles, but there are also urban areas in Virginia and Illinois that are very conservative, and many areas that are very conservative have some firearm ownership. In the North of Europe, an urban population in large cities like Dresden, which is considered by the National Rifle Association to be well-respected in the firearms industry, is the norm. In the United Kingdom, gun ownership has increased a lot in Britain since the early 1900s (and also in Europe).[14]

There are many places in America where gun ownership is common and the gun lobby is extremely influential. The NRA is the major lobby within the United States, and is strong (except when it considers violence for which it thinks it is not being represented). In this country because of this, the gun lobby is heavily centralized and in many cases there are many different sources of information available on which to draw their conclusions. It is even possible to use guns in a wide variety of situations.

Source: Wikipedia

In many of these jurisdictions, there is the police organization, which is overwhelmingly political, as well as the Gunners Rights Party. It is also very difficult to pick out many of the positions within the NRA that are supported by their membership. The most obvious position, of course, is the most effective.

In Virginia, many gun owners get a strong political opinion in their communities at large — however, in certain counties, it is not so strong in other places. In

```

Lets see how the output becomes if we turn off the top-3 neuron concepts for violent.

```markdown
Most consistent violent neurons
====================
Max possible occurrences: 10
Rank  Layer  Neuron  Occurrences  %
   1     11     197            6   60.00%
   2     11     373            5   50.00%
   3     11     680            5   50.00%
```

```markdown
In the United States, firearms are a major factor in violent crime and contributing to violent crime over and above domestic violence. However, domestic violence and other forms of violence contribute substantially to violence and are associated with disproportionate levels of domestic assault and rape. This paper reviews the relationship of domestic violence to national homicide data and concludes that regardless of the level of domestic assault, domestic violence and other forms of violence contribute substantially to the spread of violent crime.

2. Conclusion

Violence is one of the most common and violent crimes in the world. Nevertheless, the United States homicide rate is well below the rate of every other advanced capitalist nation. Moreover, domestic violence is a major cause of violence and contributes significantly to violent crime. In addition, domestic violence contributes significantly to violent crime, through its intensity and frequency. However, other factors that contribute to violence include its violence-related origins: homicide rates, population, rural areas, geographical areas, and level of educational attainment. This paper reviews the relationship of domestic violence to national homicide and provides useful information about the relationship between domestic violence and violent crime.

3. References

References

1. Conley J, Haddock P, Smith C & Deakins J , Sexual violence and homicide in the United States , Journal of Ethnicity and Migration , 44 , 4 , (469) , .

2. Sperry KJ , Violence in International Relations: How Violence Is Exploited , International Journal of International Relations , 10 , 4 , (404) , .
```

> Well, the steered generation is not bad. It means the top neurons we found do not heavily affect the generation of violence.
> Is it time to conclude that neuron tracing is not effective? There could be many reasons:
> (a) the neuron concepts are not monosematic. (b) we are tracing activations on all tokens but we should do it selectively on important generated tokens only to get a finer set of neurons.