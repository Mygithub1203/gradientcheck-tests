package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertTrue;

/**
 * Hello world!
 *
 */
public class App {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-3;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;

    public static void main( String[] args) {
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        String[] activFns = {"sigmoid","tanh","softplus"};
        boolean[] characteristic = {false,true};	//If true: run some backprop steps first

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        LossFunctions.LossFunction[] lossFunctions = {LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.MSE};
        String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        for( String afn : activFns) {
            for(boolean doLearningFirst : characteristic) {
                for(int i = 0; i<lossFunctions.length; i++) {
                    LossFunctions.LossFunction lf = lossFunctions[i];
                    String outputActivation = outputActivations[i];

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .regularization(false)
                            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                            .learningRate(1.0)
                            .seed(12345L)
                            .list()
                            .layer(0, new DenseLayer.Builder()
                                    .nIn(4).nOut(3)
                                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                    .activation(afn)
                                    .updater(Updater.SGD)
                                    .build())
                            .layer(1, new OutputLayer.Builder(lf)
                                    .activation(outputActivation)
                                    .nIn(3).nOut(3)
                                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                    .updater(Updater.SGD)
                                    .build())
                            .pretrain(false).backprop(true)
                            .build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();

                    if(doLearningFirst){
                        //Run a number of iterations of learning
                        mln.setInput(ds.getFeatures());
                        mln.setLabels(ds.getLabels());
                        mln.computeGradientAndScore();
                        double scoreBefore = mln.score();
                        for( int j = 0; j < 10; j++ )
                            mln.fit(ds);
                        mln.computeGradientAndScore();
                        double scoreAfter = mln.score();
                        //Can't test in 'characteristic mode of operation' if not learning
                        String msg = "testGradMLP2LayerIrisSimple() - score did not (sufficiently) decrease during learning - activationFn="
                                +afn+", lossFn="+lf+", outputActivation="+outputActivation+", doLearningFirst="+doLearningFirst
                                +" (before="+scoreBefore +", scoreAfter="+scoreAfter+")";
                        assertTrue(msg,scoreAfter < 0.8 *scoreBefore);
                    }

                    if( PRINT_RESULTS ){
                        System.out.println("testGradientMLP2LayerIrisSimpleRandom() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
                                +", doLearningFirst="+doLearningFirst );
                        for( int j=0; j<mln.getnLayers(); j++ ) System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

                    String msg = "testGradMLP2LayerIrisSimple() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
                            +", doLearningFirst="+doLearningFirst;
                    assertTrue(msg,gradOK);
                }
            }
        }

    }
}
