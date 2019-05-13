package it.ekursy.deeplearning4j.workshop.basic.exercises.yolo;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import it.ekursy.deeplearning4j.workshop.basic.exercises.yolo2custom.Yolo2CustomDataProvider;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.junit.Test;

public class Yolo2CustromDataProviderTest {

    @Test
    public void testFileNameToSecondsOne()
    {
        var filepath = Paths.get( "src", "main", "resources", "img_001.jpg" );

        var seconds = Yolo2CustomDataProvider.fileNameToSeconds( filepath );

        assertEquals( 0, seconds );
    }

    @Test
    public void testFileNameToSecondsBig()
    {
        var filepath = Paths.get( "src", "main", "resources", "img_5551.jpg" );

        var seconds = Yolo2CustomDataProvider.fileNameToSeconds( filepath );

        assertEquals( 5550, seconds );
    }

    @Test
    public void testSecondsToDigitsStringTwenty()
    {
        var seconds = 20;

        var digits = Yolo2CustomDataProvider.secondsToTimeDigits( seconds );

        assertArrayEquals( new String[] { "0", "0", "2", "0" }, digits.toArray( new String[ 0 ] ) );
    }

    @Test
    public void testSecondsToDigitsStringOneTwenty()
    {
        var seconds = 120;

        var digits = Yolo2CustomDataProvider.secondsToTimeDigits( seconds );

        assertArrayEquals( new String[] { "0", "2", "0", "0" }, digits.toArray( new String[ 0 ] ) );
    }

    @Test
    public void testDataInitialization() throws IOException
    {
        var dataDir = Paths.get( "src", "test", "resources", "data", "digitalclock" );
        var yoloCustomDataProvider = new Yolo2CustomDataProvider( dataDir );

        var labelMap = yoloCustomDataProvider.getLabelMap();
        assertEquals( 3, labelMap.size() );

        var path = Paths.get( "src", "test", "resources", "data", "digitalclock", "img_002.jpg" ).toAbsolutePath().toUri().toString();
        var labels = yoloCustomDataProvider.getImageObjectsForPath( path );

        assertEquals( 4, labels.size() );

        testProperLabel( labels, 0, "0" );
        testProperLabel( labels, 1, "0" );
        testProperLabel( labels, 2, "0" );
        testProperLabel( labels, 3, "1" );
    }

    private void testProperLabel(List<ImageObject> labels, int index, String value)
    {
        var imageObject = labels.get( index );
        var label = imageObject.getLabel();
        assertEquals( value, label );
    }

}
