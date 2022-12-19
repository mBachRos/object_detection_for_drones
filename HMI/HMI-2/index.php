<!DOCTYPE html>
<html lang="en">
<head>
   <title>DETECTIONS</title>
   <link rel="stylesheet" type="text/css" href="stylingtest2.css"/>  
</head>
<body>

<?php
header("refresh: 10");
?>

<br>
	<center>
	<label>LATEST DETECTION IMAGES</label>
	</center>
<br>

<center>
<l1>
<?php
echo date('H:i:s Y-m-d');
?>
</l1>
</center>

<br><br>

<l2>
                <div style="float: left">SWARM UNIT 1</div>
                <div style="float: right">SWARM UNIT 3</div>
                <div style="margin: 0 auto; width: 240px;">SWARM UNIT 2</div>
</l2>
<br>



                <div style="float: left"><?php
    echo "<img src='http://192.168.136.74/Detection_img.png' alt='SWU1' />"; 
?>
</div>
                <div style="float: right"><?php
    echo "<img src='http://192.168.136.75/Detection_img.png' alt='SWU3' />"; 
?>
</div>
                <div style="margin: 0 auto; width: 640px;"><?php
    echo "<img src='http://192.168.136.76/Detection_img.png' alt='SWU2' />"; 
?>
</div>





</body>
</html>