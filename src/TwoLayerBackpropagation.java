/**
 * Neuron is a class which represent a node in Neural Network.
 * 
 * @author      Mahmood, Mossaddeque 
 * @version     1.0
 * @since       1.0
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TwoLayerBackpropagation {

	private List<Neuron> input, hidden, output;
	private List<Pattern> pattern;
	private double learningRate, bias;
	private Random random;
	private double[] desiredValues;//desired value of output layer
	
	public TwoLayerBackpropagation(int numInput, int numHidden, int numOutput,
			int rangeMin, int rangeMax, double learningRate, Random random,
			File inputFile) {

		this.learningRate = learningRate;
		this.random = random;
		this.desiredValues = new double[numOutput];

		this.input = new ArrayList<Neuron>();
		this.hidden = new ArrayList<Neuron>();
		this.output = new ArrayList<Neuron>();
		this.pattern = readPattern(inputFile);
	
		// bias is random value between [rangeMin, rangeMax] --> [-1, 1]
		this.bias = 1;// randomDouble(rangeMin, rangeMax);
		
		// initialize inputs
		for (int i = 0; i < numInput; i++) { // set initial values to 0
			input.add(new Neuron("x" + (i + 1), 0, randomDoubleArray(numHidden, rangeMin, rangeMax))); 
		}
		// initialize hidden
		for (int i = 0; i < numHidden; i++) {
			hidden.add(new Neuron("h" + (i + 1), randomDoubleArray(numOutput, rangeMin, rangeMax)));
		}
		// initialize output
		for (int i = 0; i < numOutput; i++) {
			output.add(new Neuron("y" + (i + 1)));
		}
		
		// link inputs forward to hidden
		for (Neuron x : input) {
			x.connectLayers(hidden, Direction.NEXT_LAYER);
		}
		// link hidden
		for (Neuron h : hidden) {
			// back to inputs
			h.connectLayers(input, Direction.PREV_LAYER);
			// forward to output
			h.connectLayers(output, Direction.NEXT_LAYER);
		}
		// link output back to hidden
		for (Neuron y : output) {
			y.connectLayers(hidden, Direction.PREV_LAYER);
		}
	}
	
    public void train() {
	
		double[] error = new double[pattern.size()];
		boolean done = false;
		int epoch = 0;

		// main training loop
		while (!done) {

			// loop through input patterns, save error for each
			for (int i = 0; i < pattern.size(); i++) {

				/*** Set new pattern ***/
				setInput(pattern.get(i).values);

				/*** Feed-forward computation ***/
				forwardPass();

				/*** Backpropagation with weight updates ***/
				error[i] = backwardPass();
			}

			// increase count of epoch iterations
			epoch++;

			boolean pass = true;
			// check if error for all runs is <= 0.05
			for (int i = 0; i < error.length; i++) {
				if (error[i] > 1.05)
					pass = false;
			}

			if (pass) { // if all cases <= 0.05, convergence reached
				done = true;
			}
		}

	}
	
	private void setInput(double[] values) {
		for (int i = 0; i < values.length; i++) {
			input.get(i).setValue(values[i]);
		}
	}
    
	private double backwardPass() {

		double[] outputError = new double[output.size()];
		double[] outputDelta = new double[output.size()];
		double[] hiddenError = new double[hidden.size()];
		double[] hiddenDelta = new double[hidden.size()];

		/*** Backpropagation to the output layer ***/

		// calculate delta for output layer: d = error * sigmoid derivative
		for (int i = 0; i < output.size(); i++) {
			// error = desired - y
			outputError[i] = getOutputError(output.get(i), i);

			// using sigmoid derivative = sigmoid(v) * [1 - sigmoid(v)]
			outputDelta[i] = outputError[i] * output.get(i).getValue()
					* (1.0 - output.get(i).getValue());
		}

		/*** Backpropagation to the hidden layer ***/

		// calculate delta for hidden layer: d = error * sigmoid derivative
		for (int i = 0; i < hidden.size(); i++) {
			// error(i) = sum[outputDelta(k) * w(kj)]
			hiddenError[i] = getHiddenError(hidden.get(i), outputDelta);

			// using sigmoid derivative
			hiddenDelta[i] = hiddenError[i] * hidden.get(i).getValue()
					* (1.0 - hidden.get(i).getValue());
		}

		/*** Weight updates ***/

		// update weights connecting hidden neurons to output layer
		for (int i = 0; i < output.size(); i++) {
			for (Neuron h : output.get(i).getPrevLayer()) {
				h.getWeights()[i] += learningRate * outputDelta[i]
						* h.getValue();
			}
		}

		// update weights connecting input neurons to hidden layer
		for (int i = 0; i < hidden.size(); i++) {
			for (Neuron x : hidden.get(i).getPrevLayer()) {
				x.getWeights()[i] += learningRate * hiddenDelta[i]
						* x.getValue();
			}
		}

		// return outputError to be used when testing for convergence
		return outputError[0];
	}
	
	private void forwardPass() {

		double v, y;

		// loop through hidden layers, determine current value
		for (int i = 0; i < hidden.size(); i++) {
			v = 0;

			// get v(n) for hidden layer i
			for (Neuron x : input) {
				v += x.getWeights()[i] * x.getValue();
			}

			// add bias
			v += bias;

			// calculate f(v(n))
			y = activate(v);

			hidden.get(i).setValue(y);
		}

		// calculate output
		for (int i = 0; i < output.size(); i++) {
			v = 0;

			// get v(n) for output layer
			for (Neuron h : hidden) {
				v += h.getWeights()[i] * h.getValue();
			}

			// add bias
			v += bias;

			// calculate f(v(n))
			y = activate(v);

			output.get(i).setValue(y);
		}
	}

	private double activate(double v) {
		return (1 / (1 + Math.exp(-v))); // sigmoid function
	}

	private double getHiddenError(Neuron j, double[] outputDelta) {
		// calculate error sum[outputDelta * w(kj)]
		double sum = 0;

		for (int i = 0; i < j.getNextLayer().size(); i++) {
			sum += outputDelta[i] * j.getWeights()[i];
		}

		return sum;
	}

	private double getOutputError(Neuron k, int neuronPosition) {
		// calculate error (d - y)
		// note: desired is 1 if input contains odd # of 1's and 0 otherwise
		double sum = 0;
		double d = desiredValues[neuronPosition];
		for (Neuron x : input) {
			sum += x.getValue();
		}

		return d - k.getValue();
	}

	public void writeNetwork(String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));

			out.write("Bias: " + bias + "\n");

			out.write("\n\nInput:\n\n");
			for (Neuron x : input) {
				out.write(x.getName() + ":\n");
				for (Neuron r : x.getNextLayer()) {
					out.write("\t" + r.getName() + ", " + r.getValue() + ", "
							+ Arrays.toString(r.getWeights()) + "\n");
				}
			}

			out.write("\n\nHidden:\n\n");
			for (Neuron h : hidden) {
				out.write(h.getName() + ":\n");
				out.write("\tLeft:\n");
				for (Neuron l : h.getPrevLayer()) {
					out.write("\t" + l.getName() + ", " + l.getValue() + ", "
							+ Arrays.toString(l.getWeights()) + "\n");
				}
				out.write("\tRight:\n");
				for (Neuron r : h.getNextLayer()) {
					out.write("\t" + r.getName() + ", " + r.getValue() + "\n");
				}
			}

			out.write("\n\nOutput:\n\n");
			for (Neuron y : output) {
				out.write(y.getName() + ":\n");
				for (Neuron l : y.getPrevLayer()) {
					out.write("\t" + l.getName() + ", " + l.getValue() + ", "
							+ Arrays.toString(l.getWeights()) + "\n");
				}
			}

			out.close();

		} catch (IOException e) {
		}
	}

	private double[] randomDoubleArray(int n, double rangeMin, double rangeMax) {
		double[] a = new double[n];
		for (int i = 0; i < n; i++) {
			a[i] = randomDouble(rangeMin, rangeMax);
		}
		return a;
	}

	private double randomDouble(double rangeMin, double rangeMax) {
		return (rangeMin + (rangeMax - rangeMin) * random.nextDouble());
	}

	private List<Pattern> readPattern(File inputFile) {

		List<Pattern> p = new ArrayList<Pattern>();

		try {
			BufferedReader r = new BufferedReader(new FileReader(inputFile));
			String s = "";
			int i = 0, j = 0;
			
			while ((s = r.readLine()) != null) {
				
				i = 0;
				String[] columns = s.split(" ");
				
				double[] inputValues = new double[input.size()];
				for (i = 0; i < input.size(); i++) {
					inputValues[i] = Double.parseDouble(columns[i]);
				}
				p.add(new Pattern(inputValues));
				
				for (j = 0; j < output.size(); j++) {
					desiredValues[i] = Double.parseDouble(columns[i++]);
				}
				
			}
		} catch (IOException e) {
		}

		return p;
	}

}
