/**
 * Neuron is a class which represent a node in Neural Network.
 * 
 * @author      Mahmood, Mossaddeque 
 * @version     1.0
 * @since       1.0
 */

import java.util.ArrayList;
import java.util.List;

public class Neuron{

    private String name;
    private double value;
    private double[] weights;
    private List<Neuron> prevLayer, nextLayer;
    
    /**
     * Constructor for input layer neuron. 
     * Since input layer is the first layer of a network, no layer in the back, prevLayer is null. 
     *
     * @param   name    name of the node, e.g. for hidden layer node it may h1, h2 etc.
     * @param   value   value of the node
     * @param   weights incomming edges weights
     */    
	public Neuron(String name, double value, double[] weights) {
		this.name = name;
		this.value = value;
		this.weights = weights;
		this.prevLayer = null;
		this.nextLayer = new ArrayList<Neuron>();
	}


    /**
     * Constructor for hidden layer neuron. 
     * Since hidden layer is always in the middle layer of a network, both prevLayer and nextLayer exist. 
     * Default initial value of the node is 0.
     *
     * @param   name    name of the node, e.g. for hidden layer node it may h1, h2 etc.
     * @param   weights incomming edges weights
     */    
    public Neuron(String name, double[] weights) {  
	this.name = name;
	this.weights = weights;
	this.value = 0;
	this.prevLayer = new ArrayList<Neuron>();
	this.nextLayer = new ArrayList<Neuron>();
    }


    /**
     * Constructor for output layer neuron. 
     * Since output layer is always is the last layer of a network, nextLayer is null. 
     * Default initial value of the node is 0.
     * Output layers node do not need weights.
     *
     * @param   name    name of the node, e.g. for hidden layer node it may h1, h2 etc.
     */    
    public Neuron(String name) {  
	this.name = name;
	this.value = 0;  // default initial value
	this.prevLayer = new ArrayList<Neuron>();
	this.nextLayer = null;
    }

    /**
     * Connect nodes between layers. 
     * Since output layer is always is the last layer of a network, nextLayer is null. 
     *
     * @param   layerNeuronList name of the node, e.g. for hidden layer node it may h1, h2 etc.
     * @param   direction       in which direction nodes to be connected.
     * @return  void
     */    
	public void connectLayers(List<Neuron> layerNeuronList, Direction direction) {

		for (Neuron n : layerNeuronList) {
			if (direction == Direction.PREV_LAYER) {
				prevLayer.add(n);
			} else {
				nextLayer.add(n);
			}
		}
	}


	public final String getName() {
		return this.name;
	}

	public final void setName(final String argName) {
		this.name = argName;
	}

	public final double getValue() {
		return this.value;
	}

	public final void setValue(final double argValue) {
		this.value = argValue;
	}

	public final double[] getWeights() {
		return this.weights;
	}

	public final void setWeights(final double[] argWeights) {
		this.weights = argWeights;
	}

	public final List<Neuron> getNextLayer() {
		return this.nextLayer;
	}

	public final void setNextLayer(final List<Neuron> argNextLayer) {
		this.nextLayer = argNextLayer;
	}

	public final List<Neuron> getPrevLayer() {
		return this.prevLayer;
	}

	public final void setPrevLayer(final List<Neuron> argPrevLayer) {
		this.prevLayer = argPrevLayer;
	}


}

