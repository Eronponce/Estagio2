from django.shortcuts import render,redirect,get_object_or_404
from .models import Owner, Property, OwnerForm, PropertyForm
from django.utils.dateparse import parse_date

def controlOwner(request):
    

    owner_form = OwnerForm(request.POST or None)
    owners = Owner.objects.all()

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
        return render(request, 'owner/owner.html', {"owner_form": owner_form, "owners": owners})


    return render(request, 'owner/owner.html', {"owner_form": owner_form, "owners": owners})

def update_owner(request, id):
    print( "update"	)
    owner = get_object_or_404(Owner, id=id)
    print( "form"	)
    if request.method == 'POST':
        form = OwnerForm(request.POST, instance=owner)
        if form.is_valid():
            form.save()
            return redirect('owner_page')
    
    form = OwnerForm(instance=owner)
    return render(request, 'owner/edit.html', {'form': form})

def delete_owner(request, id):
    owner = Owner.objects.get(id=id)
    if request.method == 'POST':
        owner.delete()
        return redirect('/owner')
    return render(request, 'owner/confirm_delete.html', {'owner': owner})


def controlProperty(request):
    property_form = PropertyForm(request.POST or None)

    if request.method == 'POST':
        if property_form.is_valid():
            owner_id = property_form.cleaned_data.get('owner')
            try:
                owner = Owner.objects.get(id=owner_id)
            except Owner.DoesNotExist:
                return render(request,"property.html",{"error_message": "Proprietário não existe."})

            property_instance = property_form.save(commit=False)
            property_instance.proprietario = owner
            property_instance.save()

            return redirect('property_list')  # Redirecione para a página de lista de propriedades

    return render(request, 'owner/property.html', {"property_form": property_form, "property": Property.objects.all()})
