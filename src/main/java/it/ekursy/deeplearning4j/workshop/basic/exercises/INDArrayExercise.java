package it.ekursy.deeplearning4j.workshop.basic.exercises;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.nd4j.linalg.factory.Nd4j;

public class INDArrayExercise {

    private static final Logger LOG = LogManager.getLogger();

    public static void main(String[] args)
    {
        var zeros = Nd4j.zeros( 2, 2 );
    }
}
