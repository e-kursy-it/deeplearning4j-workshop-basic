package it.ekursy.deeplearning4j.workshop.basic.exercises;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class INDArrayExerciseTest {

    private static final double NEGATIVE_ONE = -1;

    @Test
    public void testCreateMatrixOfZeros()
    {
        var zeros = (INDArray) null;

        assertEquals( 2, zeros.rank() );
        assertEquals( 4, zeros.length() );
        assertEquals( 0, zeros.sumNumber() );
    }

    @Test
    public void testCreateZerosOnesAndTwosVstacked()
    {
        var zeros = (INDArray) null;
        var ones = (INDArray) null;
        var twos = (INDArray) null;

        var vstack = (INDArray) null;

        validateArrayShape( zeros, 2, 4, 0, List.of( 2, 2 ) );
        validateArrayShape( ones, 2, 4, 4, List.of( 2, 2 ) );
        validateArrayShape( twos, 2, 4, 8, List.of( 2, 2 ) );

        validateArrayShape( vstack, 2, 12, 12, List.of( 6, 2 ) );
    }

    @Test
    public void testCreateZerosOnesAndTwosHstacked()
    {
        var zeros = (INDArray) null;
        var ones = (INDArray) null;
        var twos = (INDArray) null;

        var hstack = (INDArray) null;

        validateArrayShape( zeros, 2, 4, 0, List.of( 2, 2 ) );
        validateArrayShape( ones, 2, 4, 4, List.of( 2, 2 ) );
        validateArrayShape( twos, 2, 4, 8, List.of( 2, 2 ) );

        validateArrayShape( hstack, 2, 12, 12, List.of( 2, 6 ) );
    }

    @Test
    public void testGetDouble()
    {
        var zeros = (INDArray) null;
        var ones = (INDArray) null;
        var twos = (INDArray) null;

        var hstack = (INDArray) null;

        validateArrayShape( zeros, 2, 4, 0, List.of( 2, 2 ) );
        validateArrayShape( ones, 2, 4, 4, List.of( 2, 2 ) );
        validateArrayShape( twos, 2, 4, 8, List.of( 2, 2 ) );

        validateArrayShape( hstack, 2, 12, 12, List.of( 2, 6 ) );

        // replace with array proper array values using getDouble
        assertEquals( 2.0, NEGATIVE_ONE, 0 );
        assertEquals( 1.0, NEGATIVE_ONE, 0 );
        assertEquals( 0.0, NEGATIVE_ONE, 0 );
    }

    @Test
    public void testPutScalar()
    {
        var zeros = Nd4j.zeros( 2, 2 );

        validateArrayShape( zeros, 2, 4, 3.0, List.of( 2, 2 ) );
        validateArrayShape( zeros.getRow( 0 ), 1, 2, 3.0, List.of( 1, 2 ) );
        validateArrayShape( zeros.getRow( 0 ), 1, 2, 0.0, List.of( 1, 2 ) );

        assertEquals( 3.0, NEGATIVE_ONE, 0 );
    }

    @Test
    public void testSumArray()
    {
        var arr = Nd4j.create( new double[][] {
            { 0.5, -0.5 },
            { 1, -1 }
        } );

        var sum = Double.MAX_VALUE;

        assertEquals( 10.2313213325, sum, 0 );
    }

    @Test
    public void testSumArray0()
    {
        var zeros = (INDArray) null;
        var ones = (INDArray) null;
        var twos = (INDArray) null;

        var vstack = (INDArray) null;

        var sum0 = vstack.sum( 0 );

        validateArrayShape( zeros, 2, 6, 0, List.of( 2, 3 ) );
        validateArrayShape( ones, 2, 6, 6, List.of( 2, 3 ) );
        validateArrayShape( twos, 2, 6, 12, List.of( 2, 3 ) );

        validateArrayShape( vstack, 2, 18, 18, List.of( 6, 3 ) );

        validateArrayShape( sum0, 2, 3, 18, List.of( 1, 3 ) );
    }

    @Test
    public void testSumArray1()
    {
        var zeros = (INDArray) null;
        var ones = (INDArray) null;
        var twos = (INDArray) null;

        var vstack = (INDArray) null;

        var sum0 = vstack.sum( 1 );

        validateArrayShape( zeros, 2, 6, 0, List.of( 2, 3 ) );
        validateArrayShape( ones, 2, 6, 6, List.of( 2, 3 ) );
        validateArrayShape( twos, 2, 6, 12, List.of( 2, 3 ) );

        validateArrayShape( vstack, 2, 18, 18, List.of( 6, 3 ) );

        validateArrayShape( sum0, 2, 6, 18, List.of( 6, 1 ) );
    }

    @Test
    public void testTransformCos()
    {
        var arr = Nd4j.create( new double[][] {
            { 0.5, -0.5 },
            { 1, -1 }
        } );

        var cosArray = (INDArray) null;

        assertEquals( 2.8357696533203125, NEGATIVE_ONE, 0 );
    }

    @Test
    public void testTransformSin()
    {
        var arr = Nd4j.create( new double[][] {
            { 0.5, -0.5 },
            { 1, -1 }
        } );

        var sinArray = (INDArray) null;

        assertEquals( 0, NEGATIVE_ONE, 0 );
    }

    @Test
    public void testReshape()
    {
        var arr = Nd4j.create( new double[][] {
            { 0.5, -0.5, 2.0 },
            { 1.0, -1.0, 3.0 }
        } );

        var reshaped = (INDArray) null;

        validateArrayShape( arr, 2, 6, 5, List.of( 2, 3 ) );
        validateArrayShape( reshaped, 1, 6, 5, List.of( 6 ) );
    }

    @Test
    public void testFindIndex()
    {
        var arr = Nd4j.create( new double[][] {
            { 0.5, -0.5 },
            { 1.0, -1.0 }
        } );


        var maxNumber = Double.MAX_VALUE;

        var maxIndex = Integer.MAX_VALUE;

        assertEquals( 1.0, maxNumber, 0 );
        assertEquals( 2, maxIndex );
    }

    @Test
    public void testBroadcast()
    {
        var arr = Nd4j.create( new double[][] {
                { 0.5, -0.5,  2.0 },
                { 1.0, -1.0,  3.0 }
        } );
        var mul = Nd4j.create( new double[][] {
                { 0.5,  0.2, -1.2 }
        } );

        var res = (INDArray) null;

        validateArrayShape( res, 2, 6, -5.550000190734863, List.of( 2, 3 ) );
    }

    private static void validateArrayShape(INDArray arr, long rank, long length, double sum, List<Integer> shapeList)
    {
        assertEquals( rank, arr.rank() );
        assertEquals( length, arr.length() );
        assertEquals( sum, arr.sumNumber() );
        var shape = shapeList.stream().mapToLong( i -> i ).toArray();
        assertArrayEquals( shape, arr.shape() );
    }
}
