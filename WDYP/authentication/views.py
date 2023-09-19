from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView, LogoutView
from .forms import UserRegisterForm

class UserLoginView(LoginView):
    template_name = 'authentication/login.html'  # Point this to your login template

class UserLogoutView(LogoutView):
    next_page = '/'  # Redirect to homepage after logout


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            # Aqui você pode adicionar mensagens de sucesso ou redirecionar para a página de login
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'authentication/register.html', {'form': form})

def home_view(request):

    return render(request, 'authentication/home.html')
