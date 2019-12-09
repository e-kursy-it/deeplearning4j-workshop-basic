package it.ekursy.deeplearning4j.workshop.basic.exercises.rest.sparkjava;

import static spark.Spark.get;
import static spark.Spark.port;
import static spark.Spark.post;
import static spark.Spark.staticFiles;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import javax.servlet.MultipartConfigElement;
import javax.servlet.ServletException;
import org.nd4j.linalg.api.ndarray.INDArray;
import spark.Request;

public class LeNetMnistExercise {

    public static void main(String[] args) throws IOException
    {
        port( 8080 );
        // read model here

        staticFiles.location( "/static/" );
        staticFiles.expireTime( 1 );

        var uploadDir = new File( "upload" );
        uploadDir.mkdir();

        staticFiles.externalLocation( "upload" );

        get( "/hello", (req, res) -> "Hello World" );

        post( "/mnist", (req, res) -> {
            res.type( "application/json" );

            var tempFile = receiveUploadedFile( uploadDir, req );

            try {
                // your implementation is here
                throw new UnsupportedOperationException();
            }
            catch ( Exception e ) {
                e.printStackTrace();
                res.status( 500 );
                return "{}";
            }
            finally {
                Files.delete( tempFile );
            }
        } );

    }

    private static int findMatchingIndex(INDArray output, double max)
    {
        throw new UnsupportedOperationException();
    }

    private static INDArray toINDArray(BufferedImage gray)
    {
        throw new UnsupportedOperationException();
    }

    private static BufferedImage resize(BufferedImage image)
    {
        var gray = new BufferedImage( 28, 28, BufferedImage.TYPE_BYTE_GRAY );

        var g = (Graphics2D) gray.getGraphics();
        g.setBackground( Color.WHITE );
        g.clearRect( 0, 0, 28, 28 );
        g.drawImage( image.getScaledInstance( 28, 28, Image.SCALE_SMOOTH ), 0, 0, null );
        g.dispose();
        return gray;
    }

    private static BufferedImage invertColors(BufferedImage inputImage)
    {
        var gray = new BufferedImage( inputImage.getWidth(), inputImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY );

        var g = (Graphics2D) gray.getGraphics();
        g.setBackground( Color.WHITE );
        g.clearRect( 0, 0, inputImage.getWidth(), inputImage.getHeight() );
        g.drawImage( inputImage, 0, 0, null );
        g.dispose();

        for (var x = 0; x < gray.getWidth(); x++) {
            for (var y = 0; y < gray.getHeight(); y++) {
                var rgba = gray.getRGB( x, y );
                var col = new Color( rgba, true );
                col = new Color( 255 - col.getRed(), 255 - col.getGreen(), 255 - col.getBlue() );
                gray.setRGB( x, y, col.getRGB() );
            }
        }
        return gray;
    }

    private static Path receiveUploadedFile(File uploadDir, Request req) throws IOException, ServletException
    {
        var tempFile = Files.createTempFile( uploadDir.toPath(), "", "" );

        req.attribute( "org.eclipse.jetty.multipartConfig", new MultipartConfigElement( "/temp" ) );

        try (var input = req.raw().getPart( "uploaded_file" ).getInputStream()) { // getPart needs to use same "name" as input field in form
            Files.copy( input, tempFile, StandardCopyOption.REPLACE_EXISTING );
        }
        return tempFile;
    }

}
