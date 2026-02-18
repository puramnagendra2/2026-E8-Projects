// Redirect to home.html when Home button is clicked
document.getElementById('home-button').onclick = function() {
    window.location.href = "{{ url_for('home') }}"; // Adjust 'home' to your route name
};