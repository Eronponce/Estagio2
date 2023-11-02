from django.shortcuts import render,redirect
from .models import Owner, Property, OwnerForm, PropertyForm
from django.utils.dateparse import parse_date
# Create your views here.
def controlOwner(request):
    

    owner_form = OwnerForm(request.POST or None)

    if request.method == 'POST':
        name = request.POST.get('name')
        cpf = request.POST.get('cpf')
        birth = request.POST.get('birth')
        owner_instance = Owner(
        name=name,
        cpf=cpf,
        birth=birth

    )
        owner_instance.save()
        owners = Owner.objects.all()
        return render(request, 'owner/owner.html', {"owner_form": owner_form, "owners": owners})


    return render(request, 'owner/owner.html', {"owner_form": owner_form})

def update_owner(request, id):
    owner = Owner.objects.get(id=id)
    owner_form = OwnerForm(request.POST or None, instance=owner)
    if owner_form.is_valid():
        owner_form.save()
        return redirect('/owner')
    return render(request, 'owner/owner.html', {'owner_form': owner_form, 'owner': owner})

def delete_owner(request, id):
    owner = Owner.objects.get(id=id)
    if request.method == 'POST':
        owner.delete()
        return redirect('/owner')
    return render(request, 'owner/confirm_delete.html', {'owner': owner})

def controlProperty(request):
    return render(request, 'owner/property.html')

