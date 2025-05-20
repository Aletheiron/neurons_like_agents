Neurons like independent agents.

What if we consider neuron like individual agent seeking its own utility?
I can go deeper a little bit and construct biology inspired simple model.
Main algorithm will be:
Firstly, we count sparkling axons in each node. Sparkling means having information ready to passing. And yeas, I use nodes for aggregation. It is not necessary but might be useful.
Then we count active dendrites. It means not connected and searching for connection.
After that every active dendrite defines the most attractive neuron. Attractiveness negatively depends on the distance and positively on the node energy. Energy is the simple sum of working connections in the given node. Theoretically energy is decaying by time
So, dendrite is attracted by a particular node and starts to move toward it. Faster or slower. At the end of timestamp coordination are renewed.
And dendrite seeks connection in the closest node if there are available sparkling axons. 
If all is ok. Dendrite tries connection, receives information, processed it by internal function. In our example function is simple and without dynamic effects like in the real neurons. Utility function is computed, if it grows connection is established, else abrupted.
Now internal utility function of the dendrite is equal to the system one. But in reality, and in general, it is not true. 
Functions are different, but for working effective systems internal utility function depends on external. And external depends on positive and negative situations that happens after internal changes. It is important. 
The next time period cycle is repeating assuming changes in the last period.
