package it.ekursy.deeplearning4j.workshop.basic.exercises.indarray;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;
import org.datavec.image.loader.NativeImageLoader;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ColorToBlackAndWhiteTest {

    static int IMAGE_WIDTH = 750;

    static int IMAGE_HEIGHT = 441;

    static double[] GRAY_FILTER = { 0.3, 0.11, 0.59 };

    static long EXPECTED_BLACK_AND_WHITE_FILE_SIZE = 204363;

    @Test
    public void colorToBlackAndWhiteTest() throws IOException
    {
        // read image
        var loader = new NativeImageLoader( IMAGE_HEIGHT, IMAGE_WIDTH, 3 );
        var image = loader.asMatrix( "src/main/resources/data/vgg16/Mackerel-tabby-750x441.jpg" );

        // make sure image fulfill criteria
        // this does not conform to give/when/then to make your life easier
        // you'd get weird errors otherwise :)
        assertEquals( 4, image.rank() );
        assertArrayEquals( new long[] { 1, 3, IMAGE_HEIGHT, IMAGE_WIDTH }, image.shape() );

        // create filter from constant above
        var bwFilter = ( INDArray) null;

        assertEquals( 2, bwFilter.rank() );
        assertEquals( GRAY_FILTER[0], bwFilter.getDouble( 0, 0 ), 0.01 );
        assertEquals( GRAY_FILTER[1], bwFilter.getDouble( 0, 1 ), 0.01 );
        assertEquals( GRAY_FILTER[2], bwFilter.getDouble( 0, 2 ), 0.01 );

        // use mulRowVector/reshape to obtain output array
        var blackAndWhiteArray = (INDArray) null;

        assertEquals( 2, blackAndWhiteArray.rank() );
        assertArrayEquals( new long[] { IMAGE_HEIGHT, IMAGE_WIDTH }, blackAndWhiteArray.shape() );

        writeImage( blackAndWhiteArray );

        var file = new File( "target/out.png" );
        assertEquals( EXPECTED_BLACK_AND_WHITE_FILE_SIZE, file.length() );
    }

    private void writeImage(INDArray bwArrSummed) throws IOException
    {
        BufferedImage bufimage = new BufferedImage( IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = (Graphics2D) bufimage.getGraphics();
        g.setBackground( Color.WHITE );
        g.clearRect( 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT );

        for (var h = 0; h < IMAGE_HEIGHT; h++) {
            for (var w = 0; w < IMAGE_WIDTH; w++) {
                var pixel = bwArrSummed.getScalar( h, w ).getInt( 0 );
                var col = new Color( pixel, pixel, pixel );
                bufimage.setRGB( w, h, col.getRGB() );
            }
        }

        ImageIO.write( bufimage, "png", new FileOutputStream( "target/out.png" ) );
    }
}
