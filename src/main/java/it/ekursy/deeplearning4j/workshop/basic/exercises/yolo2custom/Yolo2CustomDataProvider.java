package it.ekursy.deeplearning4j.workshop.basic.exercises.yolo2custom;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.jetbrains.annotations.NotNull;

public class Yolo2CustomDataProvider implements ImageObjectLabelProvider {

    private static final Logger log = LogManager.getLogger();

    public static final List<Integer[]> BOXES = new ArrayList<>() {
        {
            add( new Integer[] { 47, 39, 165, 250 } );
            add( new Integer[] { 179, 39, 293, 250 } );
            add( new Integer[] { 342, 39, 466, 250 } );
            add( new Integer[] { 478, 39, 595, 250 } );
        }
    };

    private final Map<String, List<ImageObject>> labelMap = new HashMap<>();

    public Yolo2CustomDataProvider(Path trainDir) throws IOException
    {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream( trainDir )) {

            for (Path entry : stream) {
                List<ImageObject> list = new ArrayList();

                var seconds = fileNameToSeconds( entry );
                var labels = secondsToTimeDigits( seconds );

                for (var i = 0; i < labels.size(); i++) {
                    list.add( labelToImageObject( labels, i ) );
                }
                var path = entry.toUri().toString();
                labelMap.put( pathToLabelMapKey( path ), list );
            }
        }
    }

    public static ImageObject labelToImageObject(List<String> labels, int i)
    {
        var currentLabel = labels.get( i );
        var currentBox = BOXES.get( i );

        return new ImageObject( currentBox[ 0 ], currentBox[ 1 ], currentBox[ 2 ], currentBox[ 3 ], currentLabel );
    }

    public Map<String, List<ImageObject>> getLabelMap()
    {
        return labelMap;
    }

    public static int fileNameToSeconds(Path entry)
    {
        var fileName = entry.getFileName().toString();
        fileName = fileName.replaceFirst( "[.][^.]+$", "" );
        fileName = fileName.replace( "img_", "" );
        return Integer.valueOf( fileName ) - 1;
    }

    public static List<String> secondsToTimeDigits(int seconds)
    {
        var duration = Duration.ofSeconds( seconds );
        var minutesPart = duration.toMinutesPart();
        var secondsPart = duration.toSecondsPart();

        List<String> digits = new ArrayList<>();
        appendDigits( minutesPart, digits );
        appendDigits( secondsPart, digits );

        return digits;
    }

    private static void appendDigits(int part, List<String> digits)
    {
        var partDigits = String.valueOf( part );
        if ( partDigits.length() < 1 ) {
            digits.addAll( List.of( "0", "0" ) );
        }
        else if ( partDigits.length() < 2 ) {
            digits.add( "0" );
            digits.add( partDigits.substring( 0, 1 ) );
        }
        else {
            digits.addAll( List.of( partDigits.substring( 0, 1 ), partDigits.substring( 1, 2 ) ) );
        }

    }

    @Override
    public List<ImageObject> getImageObjectsForPath(String path)
    {
        String filename = pathToLabelMapKey( path );
        return labelMap.get( filename );
    }

    public String pathToLabelMapKey(String path)
    {
        File file = new File( path );
        return file.getName();
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(URI uri)
    {
        return getImageObjectsForPath( uri.toString() );
    }
}
