import React from 'react';

const Canvas = () => {

    useLayoutEffect(() => {
        const canvas = document.getElementById("whiteboard");
        const context = canvas.getContext('2d');
        context.fillRect(10, 10, 150, 150);
        context.fillStyle="green";

        return () => {
            <canvas id="whiteboard" width="1500" height="600"></canvas>
        };
    }, [])

    // return ( <canvas id="whiteboard" width="1500" height="600"></canvas> );
}
 
export default Canvas;