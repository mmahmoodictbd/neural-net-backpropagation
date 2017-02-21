/**
 * Neuron is a class which represent a node in Neural Network.
 * 
 * @author      Mahmood, Mossaddeque 
 * @version     1.0
 * @since       1.0
 */


import java.io.File;
import java.util.Random;

public class Main{

    public static void main(String[] args) {
	Random random = new Random(1234);
	File file = new File("input.txt");
	TwoLayerBackpropagation tlb = new TwoLayerBackpropagation(4, 4, 1, -1, 1, 0.1, random, file);
	tlb.train();
	tlb.writeNetwork("output.txt");
    }

}

